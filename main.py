import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from seismic_waveform.reader import read_mseed, scan_directory, get_station_metadata
from seismic_waveform.filter import butterworth_bandpass
from seismic_waveform.arrivals import build_arrivals_dataframe, export_arrivals_csv, compute_residuals
from seismic_tomography.model import VelocityModel, Station, Event, build_sensitivity_matrix, build_travel_time_residuals
from seismic_tomography.inversion import lsqr_inversion, svd_inversion, iterative_inversion
from seismic_tomography.render import render_isosurface, render_cross_sections, render_fault_zones


def run_waveform_pipeline(args):
    print("=" * 60)
    print("  链路一: 多台站波形自动拾取与噪声压制")
    print("=" * 60)

    if args.use_sample:
        from generate_sample_data import generate_synthetic_network, generate_and_save_mseed
        print("\n[1/5] 生成合成 MiniSEED 数据...")
        filepath = generate_and_save_mseed()
        print(f"  数据已保存至: {filepath}")
        print("\n[2/5] 读取 MiniSEED 数据流...")
        st = read_mseed(filepath)
    else:
        if args.input:
            print("\n[1/5] 读取 MiniSEED 数据流...")
            if os.path.isdir(args.input):
                st = scan_directory(args.input)
            else:
                st = read_mseed(args.input)
        else:
            print("错误: 请指定输入文件/目录 (--input) 或使用样本数据 (--sample)")
            return

    print(f"  读取到 {len(st)} 条波形轨迹")
    metadata = get_station_metadata(st)
    for m in metadata[:5]:
        print(f"    {m['network']}.{m['station']}.{m['channel']} "
              f"SR={m['sampling_rate']:.0f}Hz "
              f"{m['starttime']} - {m['endtime']}")
    if len(metadata) > 5:
        print(f"    ... 共 {len(metadata)} 条元数据")

    print(f"\n[3/5] 巴特沃斯带通滤波 ({args.freqmin}-{args.freqmax} Hz, 阶数={args.order})...")
    st_filtered = butterworth_bandpass(st, freqmin=args.freqmin, freqmax=args.freqmax, order=args.order)
    print("  滤波完成")

    print(f"\n[4/5] STA/LTA 自动拾取 (STA={args.sta}s, LTA={args.lta}s)...")
    df = build_arrivals_dataframe(
        st_filtered, method=args.method,
        sta_sec=args.sta, lta_sec=args.lta,
        p_threshold=args.p_threshold, s_threshold=args.s_threshold,
    )
    print(f"  共拾取到 {len(df)} 个震相到时")
    if len(df) > 0:
        p_count = len(df[df["phase"] == "P"])
        s_count = len(df[df["phase"] == "S"])
        print(f"    P波: {p_count} 个, S波: {s_count} 个")

    df = compute_residuals(df)

    output_csv = args.output or os.path.join(config.DATA_DIR, "arrivals.csv")
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    export_arrivals_csv(df, output_csv)
    print(f"\n[5/5] 到时数据已导出至: {output_csv}")

    print("\n到时拾取结果:")
    print(df[["network", "station", "phase", "arrival_time_str", "relative_delay_ms", "residual_ms"]].to_string(index=False))

    return st_filtered, df


def run_tomography_pipeline(args):
    print("\n" + "=" * 60)
    print("  链路二: 走时层析成像与三维断层渲染")
    print("=" * 60)

    print("\n[1/4] 构建三维速度模型与台站/事件布局...")
    nx, ny, nz = args.nx, args.ny, args.nz
    model = VelocityModel(nx=nx, ny=ny, nz=nz)

    stations = []
    n_stations = 8
    for i in range(n_stations):
        angle = 2 * np.pi * i / n_stations
        stations.append(Station(
            name=f"STA{i:02d}", network="SY",
            x_km=50 + 30 * np.cos(angle),
            y_km=50 + 30 * np.sin(angle),
        ))

    np.random.seed(42)
    events = []
    n_events = 5
    for i in range(n_events):
        events.append(Event(
            x_km=30 + np.random.rand() * 40,
            y_km=30 + np.random.rand() * 40,
            z_km=5 + np.random.rand() * 30,
        ))

    print(f"  模型网格: {nx}×{ny}×{nz} = {nx * ny * nz} 个网格单元")
    print(f"  台站数: {n_stations}, 事件数: {n_events}")
    print(f"  射线数: {n_stations * n_events}")

    print("\n[2/4] 构建灵敏度矩阵 (G)...")
    G, t_calc = build_sensitivity_matrix(model, stations, events, phase="P")

    np.random.seed(42)
    n_rays = n_stations * n_events
    observed_dt = np.zeros(n_rays)
    ray_idx = 0
    for ev in events:
        for sta in stations:
            dist = np.sqrt((ev.x - sta.x) ** 2 + (ev.y - sta.y) ** 2 + (ev.z - sta.z) ** 2)
            observed_dt[ray_idx] = dist / 6.0 + np.random.randn() * 0.01
            ray_idx += 1

    residuals = observed_dt - t_calc
    print(f"  矩阵维度: {G.shape[0]} × {G.shape[1]}")
    print(f"  残差范围: [{residuals.min():.4f}, {residuals.max():.4f}] s")

    print(f"\n[3/4] 运行 {args.inv_method.upper()} 反演...")
    if args.inv_method == "lsqr":
        dm, info = lsqr_inversion(G, residuals, nx, ny, nz, args.damping, args.smoothing, args.max_iter)
        print(f"  迭代次数: {info.get('iterations', 'N/A')}")
        print(f"  残差范数: {info.get('residual_norm', 0):.6f}")
    elif args.inv_method == "svd":
        dm, info = svd_inversion(G, residuals, nx, ny, nz, args.damping)
        print(f"  使用奇异值: {info.get('n_singular_used', 'N/A')}/{info.get('total_singular', 'N/A')}")
        print(f"  方差解释: {info.get('variance_explained', 0):.2%}")
    else:
        dm, history = iterative_inversion(G, residuals, nx, ny, nz, args.damping, args.smoothing, args.max_iter)
        for h in history:
            print(f"  外迭代 {h['iteration']}: RMS残差 = {h['rms_residual']:.6f}")

    model.dv = dm
    anomaly_3d = model.get_3d_anomaly()
    print(f"  速度异常范围: [{anomaly_3d.min():.4f}, {anomaly_3d.max():.4f}]")

    print("\n[4/4] 渲染三维等值面图...")
    fig_3d = render_fault_zones(anomaly_3d, model.dx, model.dy, model.dz)
    output_html_3d = os.path.join(config.DATA_DIR, "fault_zones_3d.html")
    fig_3d.write_html(output_html_3d)
    print(f"  三维断层渲染已保存至: {output_html_3d}")

    fig_cs = render_cross_sections(anomaly_3d, model.dx, model.dy, model.dz)
    output_html_cs = os.path.join(config.DATA_DIR, "cross_sections.html")
    fig_cs.write_html(output_html_cs)
    print(f"  剖面图已保存至: {output_html_cs}")

    fig_iso = render_isosurface(anomaly_3d, model.dx, model.dy, model.dz)
    output_html_iso = os.path.join(config.DATA_DIR, "isosurface.html")
    fig_iso.write_html(output_html_iso)
    print(f"  等值面图已保存至: {output_html_iso}")

    return model, anomaly_3d


def main():
    parser = argparse.ArgumentParser(description="地震滑移反演分析系统")

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    wave_parser = subparsers.add_parser("waveform", help="波形处理与拾取")
    wave_parser.add_argument("--input", "-i", help="MiniSEED 文件或目录路径")
    wave_parser.add_argument("--output", "-o", help="到时 CSV 输出路径")
    wave_parser.add_argument("--sample", action="store_true", dest="use_sample", help="使用合成样本数据")
    wave_parser.add_argument("--freqmin", type=float, default=config.BANDPASS_FREQMIN)
    wave_parser.add_argument("--freqmax", type=float, default=config.BANDPASS_FREQMAX)
    wave_parser.add_argument("--order", type=int, default=config.BANDPASS_ORDER)
    wave_parser.add_argument("--sta", type=float, default=config.STA_WINDOW_SEC)
    wave_parser.add_argument("--lta", type=float, default=config.LTA_WINDOW_SEC)
    wave_parser.add_argument("--p-threshold", type=float, default=config.P_THRESHOLD)
    wave_parser.add_argument("--s-threshold", type=float, default=config.S_THRESHOLD)
    wave_parser.add_argument("--method", choices=["classic", "recursive"], default="classic")

    tomo_parser = subparsers.add_parser("tomography", help="走时层析成像")
    tomo_parser.add_argument("--nx", type=int, default=config.GRID_NX)
    tomo_parser.add_argument("--ny", type=int, default=config.GRID_NY)
    tomo_parser.add_argument("--nz", type=int, default=config.GRID_NZ)
    tomo_parser.add_argument("--method", dest="inv_method", choices=["lsqr", "svd", "iterative"], default="lsqr")
    tomo_parser.add_argument("--damping", type=float, default=config.DAMPING)
    tomo_parser.add_argument("--smoothing", type=float, default=config.SMOOTHING)
    tomo_parser.add_argument("--max-iter", type=int, default=config.MAX_ITERATIONS)

    dash_parser = subparsers.add_parser("dashboard", help="启动交互式面板")
    dash_parser.add_argument("--port", type=int, default=8050)
    dash_parser.add_argument("--debug", action="store_true")

    full_parser = subparsers.add_parser("full", help="运行完整流程")

    args = parser.parse_args()

    if args.command == "waveform":
        run_waveform_pipeline(args)
    elif args.command == "tomography":
        run_tomography_pipeline(args)
    elif args.command == "dashboard":
        from dashboard.app import create_app
        app = create_app()
        print(f"启动 Dash 面板: http://127.0.0.1:{args.port}")
        app.run(debug=args.debug, port=args.port)
    elif args.command == "full":
        wave_args = argparse.Namespace(
            input=None, output=None, use_sample=True,
            freqmin=config.BANDPASS_FREQMIN, freqmax=config.BANDPASS_FREQMAX,
            order=config.BANDPASS_ORDER, sta=config.STA_WINDOW_SEC, lta=config.LTA_WINDOW_SEC,
            p_threshold=config.P_THRESHOLD, s_threshold=config.S_THRESHOLD, method="classic",
        )
        run_waveform_pipeline(wave_args)

        tomo_args = argparse.Namespace(
            nx=config.GRID_NX, ny=config.GRID_NY, nz=config.GRID_NZ,
            inv_method="lsqr", damping=config.DAMPING,
            smoothing=config.SMOOTHING, max_iter=config.MAX_ITERATIONS,
        )
        run_tomography_pipeline(tomo_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
