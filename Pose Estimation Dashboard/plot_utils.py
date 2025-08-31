import plotly.graph_objects as go
import mediapipe as mp
import streamlit as st

mp_pose = mp.solutions.pose

# Create a 3D scatter+lines plot for a single frame of MediaPipe landmarks
def plot_pose_3d(results, title="3D Human Pose"):
    if not results.pose_landmarks:
        st.info("No landmarks detected for 3D visualization.")
        return go.Figure()

    lm = results.pose_landmarks.landmark
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    zs = [p.z for p in lm]

    scatter = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers",
        marker=dict(size=4, color="red"),
        name="landmarks"
    )

    conn_x, conn_y, conn_z = [], [], []
    for a, b in mp_pose.POSE_CONNECTIONS:
        conn_x += [lm[a].x, lm[b].x, None]
        conn_y += [lm[a].y, lm[b].y, None]
        conn_z += [lm[a].z, lm[b].z, None]

    lines = go.Scatter3d(
        x=conn_x, y=conn_y, z=conn_z,
        mode="lines",
        line=dict(color="blue", width=2),
        name="connections"
    )

    fig = go.Figure(data=[scatter, lines])
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        title=title
    )
    return fig

# Create a frame-by-frame Plotly animation for a list of landmark frames
def plot_pose_3d_animation(all_landmarks):
    if not all_landmarks:
        st.info("No landmarks detected for animation.")
        return

    frames = []
    for f_id, lm in enumerate(all_landmarks):
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        zs = [p.z for p in lm]

        scatter = go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(size=4, color="red"),
        )

        conn_x, conn_y, conn_z = [], [], []
        for a, b in mp_pose.POSE_CONNECTIONS:
            conn_x += [lm[a].x, lm[b].x, None]
            conn_y += [lm[a].y, lm[b].y, None]
            conn_z += [lm[a].z, lm[b].z, None]

        lines = go.Scatter3d(
            x=conn_x, y=conn_y, z=conn_z,
            mode="lines",
            line=dict(color="blue", width=2),
        )

        frames.append(go.Frame(data=[scatter, lines], name=str(f_id)))

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 50, "redraw": True},
                                  "fromcurrent": True, "mode": "immediate"}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}])
            ]
        )],
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        title="3D Body Animation"
    )

    st.plotly_chart(fig, use_container_width=True)
