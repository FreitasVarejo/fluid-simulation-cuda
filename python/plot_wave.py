# python/plot_wave.py
import argparse, numpy as np, plotly.graph_objects as go
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data", help="pasta com wave_*.dat")
    ap.add_argument("--nx", type=int, default=NX := 100)
    args = ap.parse_args()

    first = np.loadtxt(f"{args.data_dir}/wave_0000.dat")
    z0 = first[:, 2].reshape(args.nx, args.nx)

    frames = []
    for f in sorted(Path(args.data_dir).glob("wave_*.dat")):
        z = np.loadtxt(f)[:, 2].reshape(args.nx, args.nx)
        frames.append(go.Frame(data=[go.Heatmap(z=z)], name=f.stem))

    fig = go.Figure(data=[go.Heatmap(z=z0)], frames=frames)
    fig.update_layout(title="Propagação de ondas",
                      updatemenus=[{"type":"buttons",
                                    "buttons":[{"label":"Play",
                                                "method":"animate",
                                                "args":[None,{"frame":{"duration":100,"redraw":True}}]}]}])
    fig.write_html("wave_animation.html")
    print("👉  Abra wave_animation.html no navegador!")

if __name__ == "__main__":
    main()
