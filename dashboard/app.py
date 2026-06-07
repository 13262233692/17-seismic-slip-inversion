import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from obspy import UTCDateTime
import config
from seismic_waveform.reader import read_mseed, scan_directory, split_components, get_station_metadata
from seismic_waveform.filter import butterworth_bandpass
from seismic_waveform.picker import sta_lta_pick, recursive_sta_lta_pick
from seismic_waveform.arrivals import build_arrivals_dataframe, compute_residuals
from seismic_tomography.model import VelocityModel, Station, Event, build_sensitivity_matrix, build_travel_time_residuals
from seismic_tomography.inversion import lsqr_inversion, svd_inversion, iterative_inversion
from seismic_tomography.render import render_isosurface, render_cross_sections, render_fault_zones


def create_app():
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.title = "Seismic Slip Inversion"

    app.layout = html.Div([
        html.H1("地震滑移反演分析系统", style={"textAlign": "center", "fontFamily": "Microsoft YaHei"}),

        dcc.Tabs(id="main-tabs", value="waveform-tab", children=[
            dcc.Tab(label="波形处理", value="waveform-tab"),
            dcc.Tab(label="到时拾取", value="picking-tab"),
            dcc.Tab(label="层析成像", value="tomography-tab"),
            dcc.Tab(label="三维断层渲染", value="render-tab"),
        ]),

        html.Div(id="tab-content"),
    ])

    @app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
    def render_tab(tab):
        if tab == "waveform-tab":
            return _waveform_tab()
        elif tab == "picking-tab":
            return _picking_tab()
        elif tab == "tomography-tab":
            return _tomography_tab()
        elif tab == "render-tab":
            return _render_tab()
        return html.Div()

    def _waveform_tab():
        return html.Div([
            html.H3("多台站波形处理"),
            html.Div([
                html.Label("数据目录:"),
                dcc.Input(id="data-dir", value=config.DATA_DIR, style={"width": "300px"}),
                html.Button("加载数据", id="load-btn", n_clicks=0),
            ], style={"marginBottom": "20px"}),
            html.Div([
                html.Label("带通滤波低频 (Hz):"),
                dcc.Input(id="freqmin", type="number", value=config.BANDPASS_FREQMIN, style={"width": "80px"}),
                html.Label("高频 (Hz):"),
                dcc.Input(id="freqmax", type="number", value=config.BANDPASS_FREQMAX, style={"width": "80px"}),
                html.Label("阶数:"),
                dcc.Input(id="filter-order", type="number", value=config.BANDPASS_ORDER, style={"width": "60px"}),
                html.Button("应用滤波", id="filter-btn", n_clicks=0),
            ], style={"marginBottom": "20px"}),
            html.Div([
                html.Button("使用样本数据", id="sample-btn", n_clicks=0),
            ], style={"marginBottom": "20px"}),
            dcc.Graph(id="waveform-graph"),
            html.Div(id="waveform-info"),
        ])

    def _picking_tab():
        return html.Div([
            html.H3("STA/LTA 自动拾取"),
            html.Div([
                html.Label("STA 窗口 (s):"),
                dcc.Input(id="sta-window", type="number", value=config.STA_WINDOW_SEC, style={"width": "80px"}),
                html.Label("LTA 窗口 (s):"),
                dcc.Input(id="lta-window", type="number", value=config.LTA_WINDOW_SEC, style={"width": "80px"}),
                html.Label("P 阈值:"),
                dcc.Input(id="p-threshold", type="number", value=config.P_THRESHOLD, style={"width": "80px"}),
                html.Label("S 阈值:"),
                dcc.Input(id="s-threshold", type="number", value=config.S_THRESHOLD, style={"width": "80px"}),
            ], style={"marginBottom": "10px"}),
            html.Div([
                html.Label("拾取方法:"),
                dcc.Dropdown(id="pick-method", options=[
                    {"label": "Classic STA/LTA", "value": "classic"},
                    {"label": "Recursive STA/LTA", "value": "recursive"},
                ], value="classic", style={"width": "200px"}),
                html.Button("开始拾取", id="pick-btn", n_clicks=0),
            ], style={"marginBottom": "20px"}),
            dcc.Graph(id="picking-graph"),
            html.Div(id="arrivals-table-container"),
        ])

    def _tomography_tab():
        return html.Div([
            html.H3("走时层析成像反演"),
            html.Div([
                html.Label("反演方法:"),
                dcc.Dropdown(id="inversion-method", options=[
                    {"label": "LSQR", "value": "lsqr"},
                    {"label": "SVD", "value": "svd"},
                    {"label": "Iterative LSQR", "value": "iterative"},
                ], value="lsqr", style={"width": "200px"}),
            ], style={"marginBottom": "10px"}),
            html.Div([
                html.Label("阻尼系数:"),
                dcc.Input(id="damping", type="number", value=config.DAMPING, style={"width": "100px"}),
                html.Label("平滑系数:"),
                dcc.Input(id="smoothing", type="number", value=config.SMOOTHING, style={"width": "100px"}),
                html.Label("最大迭代:"),
                dcc.Input(id="max-iter", type="number", value=config.MAX_ITERATIONS, style={"width": "80px"}),
            ], style={"marginBottom": "10px"}),
            html.Div([
                html.Label("网格 NX:"),
                dcc.Input(id="grid-nx", type="number", value=config.GRID_NX, style={"width": "60px"}),
                html.Label("NY:"),
                dcc.Input(id="grid-ny", type="number", value=config.GRID_NY, style={"width": "60px"}),
                html.Label("NZ:"),
                dcc.Input(id="grid-nz", type="number", value=config.GRID_NZ, style={"width": "60px"}),
            ], style={"marginBottom": "10px"}),
            html.Button("运行反演", id="invert-btn", n_clicks=0),
            html.Button("使用样本数据运行", id="invert-sample-btn", n_clicks=0),
            dcc.Graph(id="tomography-graph"),
            html.Div(id="inversion-info"),
        ])

    def _render_tab():
        return html.Div([
            html.H3("三维断层渲染"),
            html.Div([
                html.Label("低速异常阈值:"),
                dcc.Input(id="iso-low", type="number", value=config.ISOVALUE_LOW, step=0.05, style={"width": "100px"}),
                html.Label("高速异常阈值:"),
                dcc.Input(id="iso-high", type="number", value=config.ISOVALUE_HIGH, step=0.05, style={"width": "100px"}),
            ], style={"marginBottom": "10px"}),
            html.Button("渲染等值面", id="render-btn", n_clicks=0),
            dcc.Graph(id="isosurface-graph"),
        ])

    _app_data = {"st": None, "st_filtered": None, "arrivals_df": None, "model": None, "anomaly": None}

    @app.callback(
        [Output("waveform-graph", "figure"), Output("waveform-info", "children")],
        [Input("load-btn", "n_clicks"), Input("filter-btn", "n_clicks"), Input("sample-btn", "n_clicks")],
        [State("data-dir", "value"), State("freqmin", "value"), State("freqmax", "value"), State("filter-order", "value")],
    )
    def update_waveform(load_clicks, filter_clicks, sample_clicks, data_dir, freqmin, freqmax, order):
        ctx = dash.callback_context
        if not ctx.triggered:
            return go.Figure(), ""

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "sample-btn":
            from generate_sample_data import generate_synthetic_network
            _app_data["st"] = generate_synthetic_network()
            _app_data["st_filtered"] = None

        elif trigger_id == "load-btn" and data_dir:
            try:
                _app_data["st"] = scan_directory(data_dir)
                _app_data["st_filtered"] = None
            except Exception as e:
                return go.Figure(), f"加载失败: {str(e)}"

        elif trigger_id == "filter-btn":
            if _app_data["st"] is not None:
                _app_data["st_filtered"] = butterworth_bandpass(
                    _app_data["st"], freqmin=freqmin, freqmax=freqmax, order=order
                )

        st_display = _app_data.get("st_filtered") or _app_data.get("st")
        if st_display is None:
            return go.Figure(), "请先加载数据或使用样本数据"

        fig = _plot_waveforms(st_display)
        info = f"共 {len(st_display)} 条波形轨迹"
        if _app_data.get("st_filtered") is not None:
            info += f" (已滤波 {freqmin}-{freqmax} Hz, 阶数 {order})"

        return fig, info

    @app.callback(
        [Output("picking-graph", "figure"), Output("arrivals-table-container", "children")],
        [Input("pick-btn", "n_clicks")],
        [State("sta-window", "value"), State("lta-window", "value"),
         State("p-threshold", "value"), State("s-threshold", "value"), State("pick-method", "value")],
    )
    def run_picking(clicks, sta_w, lta_w, p_thr, s_thr, method):
        st = _app_data.get("st_filtered") or _app_data.get("st")
        if st is None:
            return go.Figure(), "请先加载并滤波波形数据"

        df = build_arrivals_dataframe(
            st, method=method, sta_sec=sta_w, lta_sec=lta_w,
            p_threshold=p_thr, s_threshold=s_thr,
        )
        _app_data["arrivals_df"] = df

        fig = _plot_picks(st, df)

        if len(df) > 0:
            table = dash_table.DataTable(
                data=df[["network", "station", "phase", "arrival_time_str", "relative_delay_ms"]].to_dict("records"),
                columns=[
                    {"name": "台网", "id": "network"},
                    {"name": "台站", "id": "station"},
                    {"name": "震相", "id": "phase"},
                    {"name": "到时", "id": "arrival_time_str"},
                    {"name": "相对延迟 (ms)", "id": "relative_delay_ms"},
                ],
                page_size=20,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "center", "fontFamily": "Microsoft YaHei"},
            )
        else:
            table = html.P("未检测到震相到时，请调整阈值")

        return fig, table

    @app.callback(
        [Output("tomography-graph", "figure"), Output("inversion-info", "children")],
        [Input("invert-btn", "n_clicks"), Input("invert-sample-btn", "n_clicks")],
        [State("inversion-method", "value"), State("damping", "value"),
         State("smoothing", "value"), State("max-iter", "value"),
         State("grid-nx", "value"), State("grid-ny", "value"), State("grid-nz", "value")],
    )
    def run_inversion(inv_clicks, sample_clicks, method, damping, smoothing, max_iter, nx, ny, nz):
        ctx = dash.callback_context
        if not ctx.triggered:
            return go.Figure(), ""

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        stations, events, observed_dt = _generate_tomography_sample(nx, ny, nz)

        model = VelocityModel(nx=nx, ny=ny, nz=nz)
        _app_data["model"] = model

        G, t_calc = build_sensitivity_matrix(model, stations, events, phase="P")
        residuals = observed_dt - t_calc

        if method == "lsqr":
            dm, info = lsqr_inversion(G, residuals, nx, ny, nz, damping, smoothing, max_iter)
        elif method == "svd":
            dm, info = svd_inversion(G, residuals, nx, ny, nz, damping)
        else:
            dm, history = iterative_inversion(G, residuals, nx, ny, nz, damping, smoothing, max_iter)
            info = history[-1]["info"] if history else {}

        model.dv = dm
        anomaly_3d = model.get_3d_anomaly()
        _app_data["anomaly"] = anomaly_3d

        fig = render_cross_sections(anomaly_3d, model.dx, model.dy, model.dz)

        info_text = f"反演方法: {method} | 迭代次数: {info.get('iterations', 'N/A')} | 残差范数: {info.get('residual_norm', 0):.4f}"
        if "n_singular_used" in info:
            info_text += f" | 奇异值使用: {info['n_singular_used']}/{info['total_singular']}"
            info_text += f" | 方差解释: {info.get('variance_explained', 0):.2%}"

        return fig, html.P(info_text)

    @app.callback(
        Output("isosurface-graph", "figure"),
        [Input("render-btn", "n_clicks")],
        [State("iso-low", "value"), State("iso-high", "value")],
    )
    def render_3d(clicks, iso_low, iso_high):
        anomaly = _app_data.get("anomaly")
        model = _app_data.get("model")

        if anomaly is None or model is None:
            return go.Figure()

        fig = render_fault_zones(
            anomaly, model.dx, model.dy, model.dz,
            fault_threshold=iso_low,
        )
        return fig

    def _plot_waveforms(st):
        fig = make_subplots(
            rows=min(len(st), 12), cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
        )

        traces_to_show = list(st[:12])
        for i, tr in enumerate(traces_to_show):
            t = np.arange(tr.stats.npts) / tr.stats.sampling_rate
            fig.add_trace(
                go.Scattergl(
                    x=t, y=tr.data,
                    name=f"{tr.stats.station}.{tr.stats.channel}",
                    line=dict(width=0.5),
                ),
                row=i + 1, col=1,
            )

        fig.update_layout(
            height=max(300, len(traces_to_show) * 80),
            showlegend=False,
            title_text="波形显示",
        )
        return fig

    def _plot_picks(st, df):
        fig = make_subplots(
            rows=min(len(st), 8), cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
        )

        traces_to_show = list(st[:8])
        for i, tr in enumerate(traces_to_show):
            t = np.arange(tr.stats.npts) / tr.stats.sampling_rate
            fig.add_trace(
                go.Scattergl(
                    x=t, y=tr.data,
                    name=f"{tr.stats.station}.{tr.stats.channel}",
                    line=dict(width=0.5),
                ),
                row=i + 1, col=1,
            )

            sta_picks = df[
                (df["station"] == tr.stats.station) & (df["phase"] == "P")
            ]
            for _, pick in sta_picks.iterrows():
                pick_t = (pick["arrival_time"] - tr.stats.starttime)
                fig.add_vline(
                    x=pick_t, line_width=2, line_dash="dash", line_color="red",
                    row=i + 1, col=1,
                )

            sta_picks_s = df[
                (df["station"] == tr.stats.station) & (df["phase"] == "S")
            ]
            for _, pick in sta_picks_s.iterrows():
                pick_t = (pick["arrival_time"] - tr.stats.starttime)
                fig.add_vline(
                    x=pick_t, line_width=2, line_dash="dash", line_color="blue",
                    row=i + 1, col=1,
                )

        fig.update_layout(
            height=max(300, len(traces_to_show) * 100),
            showlegend=False,
            title_text="P波(红线) / S波(蓝线) 拾取结果",
        )
        return fig

    def _generate_tomography_sample(nx, ny, nz):
        np.random.seed(42)
        n_stations = 8
        n_events = 5

        stations = []
        for i in range(n_stations):
            angle = 2 * np.pi * i / n_stations
            stations.append(Station(
                name=f"STA{i:02d}", network="SY",
                x_km=50 + 30 * np.cos(angle),
                y_km=50 + 30 * np.sin(angle),
            ))

        events = []
        for i in range(n_events):
            events.append(Event(
                x_km=30 + np.random.rand() * 40,
                y_km=30 + np.random.rand() * 40,
                z_km=5 + np.random.rand() * 30,
            ))

        n_rays = n_events * n_stations
        observed_dt = np.zeros(n_rays)
        ray_idx = 0
        for ev in events:
            for sta in stations:
                dist = np.sqrt((ev.x - sta.x) ** 2 + (ev.y - sta.y) ** 2 + (ev.z - sta.z) ** 2)
                observed_dt[ray_idx] = dist / 6.0
                observed_dt[ray_idx] += np.random.randn() * 0.01
                ray_idx += 1

        return stations, events, observed_dt

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=8050)
