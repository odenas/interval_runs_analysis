
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from bspline_model import *
from workout import *



def create_date_selector_dashboard(analytics_list):
    """
    analytics_list: A list of TreadmillAnalytic objects, one per date.
    """
    template = "plotly"
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,  vertical_spacing=0.1, x_title="Time (Seconds)")
    
    # Store how many traces we add per date (e.g., Raw + fit + fit band + knots + accel + accel band = 6)
    traces_per_date = 6
    menu_buttons = []
    
    for i, analytic in enumerate(analytics_list):
        # 1. Extract Data
        time = analytic.time
        hr = analytic.hr
        assert time.shape[0] == hr.shape[0]
        hr_stats = MedianHdiSample(np.median(analytic.post_mu, axis=0), az.hdi(analytic.trace, var_names="mu", hdi_prob=0.89).mu.values)
        # hr_stats = MedianHdiSample.from_samples(analytic.post_mu, analytic.post_mu.shape[0])
        mu_hdi = hr_stats.hdi
        acc_stats = MedianHdiSample(np.median(analytic.post_accel, axis=0), az.hdi(analytic.trace, var_names="accel", hdi_prob=0.89).accel.values)
        # acc_stats = MedianHdiSample.from_samples(analytic.post_accel, analytic.post_accel.shape[0])
        acc_hdi = acc_stats.hdi
        knots = analytic.model.knots.flatten()
        
        # 2. Add Traces (Set visible=True only for the first date)
        is_visible = (i == 0)
        
        # Trace A: Raw Data
        fig.add_trace(go.Scatter(
            x=time, y=hr, mode='markers', 
            name=f'Raw {analytic.session_id}',
            marker=dict(color='gray', opacity=0.2),
            visible=is_visible
        ), row=1, col=1)
        
        # Trace B: Smoothed Bayesian Fit
        fig.add_trace(go.Scatter(
            x=time, y=hr_stats.median, mode='lines', 
            name=f'Fit {analytic.session_id}',
            line=dict(color='rgba(43, 137, 180, 1)', width=1),
            visible=is_visible
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([mu_hdi[:, 1], mu_hdi[::-1, 0]]),
            fill='toself', fillcolor='rgba(43, 137, 180, 0.35)',
            line=dict(color='rgba(255,255,255,0)'),
            name="HR 89% HDI", visible=is_visible, hoverinfo="skip"
        ), row=1, col=1)

        # Trace C: Rug for knots
        fig.add_trace(go.Scatter(
            x=knots, y=[hr_stats.median.max() - 5] * len(knots), 
            mode='markers', name="Knots",
            marker=dict(
                symbol='line-ns-open', # This creates the 'rug' tick shape
                size=16,               # Height of the rug tick
                line=dict(width=2, color="black")
            ),
            visible=is_visible
        ), row=1, col=1)

        # lower subplot: acceleration
        fig.add_trace(go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([acc_hdi[:, 1], acc_hdi[::-1, 0]]),
            fill='toself', fillcolor='rgba(43, 137, 180, 0.35)',
            line=dict(color='rgba(255,255,255,0)'),
            name="Accel 89% HDI", visible=is_visible, hoverinfo="skip"
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=time, y=acc_stats.median, mode='lines', 
            name=f'Accel {analytic.session_id}',
            line=dict(color='rgba(43, 137, 180, 1)', width=2),
            visible=is_visible
        ), row=2, col=1)


        # 3. Create visibility mask for this specific date
        # If we have 4 dates, mask looks like: [False, False, True, True, False, False, False, False]
        visibility_mask = [False] * (len(analytics_list) * traces_per_date)
        visibility_mask[i * traces_per_date : (i + 1) * traces_per_date] = [True] * traces_per_date
        
        # 4. Create the button for this date
        menu_buttons.append(dict(
            label=str(analytic.session_id),
            method="update",
            args=[{"visible": visibility_mask}, {"title": f"Session Analysis: {analytic.session_id}"}]
        ))

    # 5. Add the Menu to the Layout
    fig.update_layout(
        updatemenus=[dict(
            active=0,buttons=menu_buttons,
            x=1.0, y=1.15, xanchor="right", yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.8)", # Slight transparency
            bordercolor="gray", borderwidth=1
        )],
        # Title adjustment to ensure it doesn't clash
        title=dict(
            text=f"Session Analysis: {analytics_list[0].session_id}",
            x=0.05,          # Keep the title on the far left
            xanchor="left"
        ),
        height=600, template="plotly",
        hovermode="x unified"
    )
    # Label the Top Subplot (Row 1)
    fig.update_yaxes(title_text="Heart Rate (BPM)", row=1, col=1)

    # Label the Bottom Subplot (Row 2)
    fig.update_yaxes(title_text="Acceleration (BPM/s)", row=2, col=1)

    # Optional: Set fixed ranges if you want to keep the scale consistent 
    # while switching between sessions in the dropdown
    # fig.update_yaxes(range=[60, 200], row=1, col=1) # Standard HR range
    # fig.update_yaxes(range=[-1, 3], row=2, col=1)   # Standard Accel range
    
    return fig
