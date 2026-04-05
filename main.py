"""
main.py
WILDFIRE simulation — entry point.

Visual features:
  - Truck rendered as an oriented rectangle with direction arrow
  - Planned Dubins path drawn as a dashed line on the field
  - Wumpus path drawn faintly
  - Fire cells pulse brighter as they approach spread time
  - Blue ring shows extinguish radius when truck is dwelling
  - Extinguish progress bar next to truck
  - Hunt mode: truck turns red, chases Wumpus
  - Full stats panel with scores, counts, CPU times

Saves (same folder as main.py):
  wildfire_scores.png, wildfire_cpu.png, wildfire_env.png, wildfire_anim.gif

Usage:
  python main.py              <- live popup ON
  python main.py --no-display <- headless

RBE-550 Motion Planning — Assignment 5: WILDFIRE
Samuel Oluwakorede Oyefusi | WPI | Spring 2026
"""

import sys, os, time, math
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

_SHOW = '--no-display' not in sys.argv

def _setup_backend():
    import matplotlib
    if not _SHOW:
        matplotlib.use('Agg'); return
    for b in ('Qt5Agg','TkAgg','WXAgg','MacOSX','Agg'):
        try:
            matplotlib.use(b)
            import matplotlib.pyplot as _p; _p.figure(); _p.close('all'); return
        except Exception: continue

_setup_backend()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch

from config import (SEEDS, N_RUNS, GRID_N, CELL_SIZE, FIELD_SIZE,
                    SIM_DURATION, DT, VIZ_INTERVAL, VIZ_PAUSE,
                    SNAP_EVERY, ANIM_FPS, EXTINGUISH_R, FIRE_SPREAD_T,
                    TRUCK_L, TRUCK_W)
from agents import (State, generate_grid, ObstacleManager, Wumpus, Firetruck)
from sampling_planner import PRMPlanner

# ── Colour palette ────────────────────────────────────────────────────────────
_COL = {
    State.INTACT:       (0.18, 0.42, 0.18),
    State.BURNING:      (0.91, 0.27, 0.04),
    State.EXTINGUISHED: (0.13, 0.46, 0.68),
    State.BURNED:       (0.22, 0.22, 0.22),
}
_COL_FLASH   = (1.00, 0.90, 0.10)   # newly ignited flash (yellow)
_COL_URGENT  = (1.00, 0.55, 0.05)   # fire about to spread (brighter)
_BG          = '#1a1a1a'
_C_WMP       = '#cc44ff'
_C_TRUCK     = '#ff8800'
_C_TRUCK_HUNT= '#ff2222'
_C_PATH      = '#ffcc44'
_C_WMP_PATH  = '#aa66ff'


def _rgb(c): return c if isinstance(c,tuple) else tuple(int(c.lstrip('#')[i:i+2],16)/255 for i in (0,2,4))


# ─────────────────────────────────────────────────────────────────────────────
# Live display
# ─────────────────────────────────────────────────────────────────────────────

class LiveDisplay:
    """
    Two-panel interactive window.
    Left : field map  |  Right : stats panel
    """

    def __init__(self, run_idx, seed, n_obstacles):
        plt.ion()
        self.fig = plt.figure(figsize=(16, 8), facecolor='#0d0d0d')
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title(f'WILDFIRE — Run {run_idx+1} / {N_RUNS}')

        # GridSpec: field takes 60%, stats 40%
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.55, 1],
                                   wspace=0.04,
                                   left=0.04, right=0.98, top=0.94, bottom=0.06)
        self._af = self.fig.add_subplot(gs[0])
        self._ai = self.fig.add_subplot(gs[1])

        self.fig.suptitle(
            f'WILDFIRE  ·  Run {run_idx+1}  ·  seed {seed}',
            color='#ff8800', fontsize=15, fontweight='bold', y=0.98)

        # ── Field panel ───────────────────────────────────────────────────────
        af = self._af
        af.set_facecolor(_BG)
        af.set_xlim(-2, FIELD_SIZE+2); af.set_ylim(-2, FIELD_SIZE+2)
        af.set_aspect('equal')
        af.set_xlabel('x (m)', color='#888', fontsize=9)
        af.set_ylabel('y (m)', color='#888', fontsize=9)
        af.tick_params(colors='#555', labelsize=8)
        for sp in af.spines.values(): sp.set_edgecolor('#333')

        # Grid image (RGBA for alpha channel)
        self._img_data = np.zeros((GRID_N, GRID_N, 4), dtype=np.float32)
        self._img = af.imshow(
            self._img_data, origin='lower',
            extent=[0, FIELD_SIZE, 0, FIELD_SIZE],
            interpolation='nearest', aspect='auto', zorder=2)

        # Wumpus path (faint purple line)
        self._wpath, = af.plot([], [], '-', color=_C_WMP_PATH,
                               lw=1.0, alpha=0.35, zorder=4)

        # Truck planned path (dashed yellow)
        self._tpath, = af.plot([], [], '--', color=_C_PATH,
                               lw=1.6, alpha=0.75, zorder=5, dashes=(6,4))

        # Extinguish radius ring (shown when dwelling)
        self._ext_ring = plt.Circle((0,0), EXTINGUISH_R, fill=False,
                                     edgecolor='#44aaff', lw=1.5,
                                     alpha=0, zorder=6)
        af.add_patch(self._ext_ring)

        # Extinguish progress arc (shows as a sweeping arc)
        self._ext_arc = mpatches.Wedge((0,0), EXTINGUISH_R, 90, 90,
                                        color='#44aaff', alpha=0, zorder=7)
        af.add_patch(self._ext_arc)

        # Truck body (polygon — oriented rectangle)
        self._truck_body = plt.Polygon(
            [[0,0],[0,0],[0,0],[0,0]], closed=True,
            color=_C_TRUCK, zorder=10, linewidth=1.2, edgecolor='white')
        af.add_patch(self._truck_body)

        # Truck direction arrow
        self._truck_arrow, = af.plot([], [], '-', color='white',
                                      lw=2.0, zorder=11)

        # Wumpus marker
        self._wumpus_dot, = af.plot([], [], 's', ms=12, color=_C_WMP,
                                     markeredgecolor='white',
                                     markeredgewidth=1.0, zorder=10)

        # Fire spread radius preview (ring on oldest burning cell)
        self._spread_ring = plt.Circle((0,0), 0, fill=False,
                                        edgecolor='#ff4400', lw=0.8,
                                        alpha=0, linestyle=':', zorder=3)
        af.add_patch(self._spread_ring)

        # Time label on field
        self._field_t = af.text(2, FIELD_SIZE-8, 't = 0 s',
                                 color='#cccccc', fontsize=10,
                                 fontweight='bold', zorder=15)

        # Legend
        af.legend(handles=[
            mpatches.Patch(color=_COL[State.INTACT],       label='Intact'),
            mpatches.Patch(color=_COL[State.BURNING],      label='Burning'),
            mpatches.Patch(color=_COL[State.EXTINGUISHED], label='Extinguished'),
            mpatches.Patch(color=_COL[State.BURNED],       label='Burned'),
            plt.Line2D([0],[0], marker='s', color='w',
                       markerfacecolor=_C_WMP, ms=9, label='Wumpus'),
            mpatches.Patch(color=_C_TRUCK, label='Firetruck'),
        ], loc='upper right', fontsize=7.5,
           facecolor='#1a1a1a', labelcolor='#cccccc',
           edgecolor='#444', framealpha=0.85)

        # ── Stats panel ───────────────────────────────────────────────────────
        ai = self._ai
        ai.set_facecolor('#111111'); ai.set_xlim(0,1); ai.set_ylim(0,1)
        ai.axis('off')

        def _txt(x,y,s,**kw):
            return ai.text(x,y,s,transform=ai.transAxes,
                           fontfamily='monospace',**kw)

        ai.text(0.5,0.97,'WILDFIRE',color='#ff8800',fontsize=24,
                fontweight='bold',ha='center',va='top',
                transform=ai.transAxes)
        ai.text(0.5,0.90,'RBE-550 Motion Planning  ·  WPI  ·  Spring 2026',
                color='#666',fontsize=8.5,ha='center',va='top',
                transform=ai.transAxes)
        ai.text(0.5,0.85,'Samuel Oluwakorede Oyefusi',
                color='#888',fontsize=8,ha='center',va='top',
                transform=ai.transAxes)

        # Divider
        ai.plot([0, 1], [0.82, 0.82], color='#333', lw=0.8, transform=ai.transAxes)

        kw = dict(va='top', fontsize=11, color='#cccccc')

        self._t_time  = _txt(0.04, 0.79, 'Time:         0 s', **kw)
        self._t_mode  = _txt(0.04, 0.73, 'Mode:  firefighting', color='#44ff88',
                              fontsize=10, va='top')

        ai.plot([0, 1], [0.70, 0.70], color='#333', lw=0.6, transform=ai.transAxes)

        # Wumpus score box
        ai.add_patch(mpatches.FancyBboxPatch(
            (0.03,0.56), 0.44, 0.12, boxstyle='round,pad=0.01',
            facecolor='#2a1a2a', edgecolor=_C_WMP, lw=1.2,
            transform=ai.transAxes))
        ai.text(0.25,0.67,'WUMPUS',color=_C_WMP,fontsize=9,
                fontweight='bold',ha='center',va='top',
                transform=ai.transAxes)
        self._t_ws = ai.text(0.25,0.62,'0 pts',color='white',fontsize=14,
                              fontweight='bold',ha='center',va='top',
                              transform=ai.transAxes)

        # Truck score box
        ai.add_patch(mpatches.FancyBboxPatch(
            (0.53,0.56), 0.44, 0.12, boxstyle='round,pad=0.01',
            facecolor='#2a1a0a', edgecolor=_C_TRUCK, lw=1.2,
            transform=ai.transAxes))
        ai.text(0.75,0.67,'FIRETRUCK',color=_C_TRUCK,fontsize=9,
                fontweight='bold',ha='center',va='top',
                transform=ai.transAxes)
        self._t_ts = ai.text(0.75,0.62,'0 pts',color='white',fontsize=14,
                              fontweight='bold',ha='center',va='top',
                              transform=ai.transAxes)

        ai.plot([0, 1], [0.54, 0.54], color='#333', lw=0.6, transform=ai.transAxes)

        # Obstacle counts
        kw2 = dict(va='top', fontsize=10, color='#aaaaaa')
        self._t_int  = _txt(0.04,0.51,f'Intact:       {n_obstacles}',**kw2)
        self._t_burn = _txt(0.04,0.45,'Burning:      0', color='#ff6622',
                             fontsize=10,va='top')
        self._t_ext  = _txt(0.04,0.39,'Extinguished: 0', color='#44aaff',
                             fontsize=10,va='top')
        self._t_burned=_txt(0.04,0.33,'Burned:       0', color='#888888',
                             fontsize=10,va='top')

        ai.plot([0, 1], [0.30, 0.30], color='#333', lw=0.6, transform=ai.transAxes)

        # CPU stats
        kw3 = dict(va='top', fontsize=9, color='#777777')
        ai.text(0.04,0.27,'CPU planning time:',**kw3)
        self._t_wcpu = _txt(0.04,0.22,'  Wumpus A*: 0.000 s',**kw3)
        self._t_tcpu = _txt(0.04,0.17,'  Truck PRM: 0.000 s (build + query)',**kw3)

        ai.plot([0, 1], [0.13, 0.13], color='#333', lw=0.6, transform=ai.transAxes)

        # Progress bar
        ai.add_patch(mpatches.FancyBboxPatch(
            (0.04,0.05), 0.92, 0.055, boxstyle='round,pad=0.01',
            facecolor='#222', edgecolor='#444', lw=0.8,
            transform=ai.transAxes, zorder=1))
        self._prog_bar = mpatches.FancyBboxPatch(
            (0.04,0.05), 0.0, 0.055, boxstyle='round,pad=0.01',
            facecolor='#ff8800', edgecolor='none',
            transform=ai.transAxes, zorder=2)
        ai.add_patch(self._prog_bar)
        self._t_pct = ai.text(0.50,0.075,'0%',ha='center',va='center',
                               fontsize=9,color='white',fontweight='bold',
                               transform=ai.transAxes,zorder=3)

        self.fig.canvas.draw()
        plt.pause(VIZ_PAUSE)

    # ── update ────────────────────────────────────────────────────────────────

    def update(self, obs, wumpus, truck, sim_time):
        # ── Field image ───────────────────────────────────────────────────────
        img = np.zeros((GRID_N, GRID_N, 4), dtype=np.float32)
        for (r,c), s in obs.state.items():
            base = np.array(_COL[s] + (1.0,))
            if s == State.BURNING:
                age   = obs.burn_age(r, c, sim_time)
                t_left = obs.time_to_spread(r, c, sim_time)
                # Flash on newly ignited
                if (r,c) in obs.flash:
                    alpha_flash = obs.flash[(r,c)] / 8.0
                    base[:3] = np.array(_COL_FLASH) * alpha_flash + base[:3] * (1-alpha_flash)
                elif t_left < 3.0:
                    # Pulse brighter when about to spread
                    pulse = 0.5 + 0.5 * math.sin(sim_time * 6)
                    bright = np.array(_COL_URGENT)
                    base[:3] = bright * pulse + np.array(_COL[State.BURNING]) * (1-pulse)
            img[r,c] = base

        self._img.set_data(img)

        # ── Truck body ────────────────────────────────────────────────────────
        tx,ty,th = truck.pos
        cos_t,sin_t = math.cos(th), math.sin(th)
        hw,hl = TRUCK_W/2, TRUCK_L/2
        corners = []
        for dx,dy in [(-hl,-hw),(hl,-hw),(hl,hw),(-hl,hw)]:
            corners.append([tx+dx*cos_t-dy*sin_t, ty+dx*sin_t+dy*cos_t])
        self._truck_body.set_xy(corners)
        truck_col = _C_TRUCK_HUNT if truck.hunt_mode else _C_TRUCK
        self._truck_body.set_facecolor(truck_col)

        # Direction arrow
        arr_len = hl * 1.4
        self._truck_arrow.set_data(
            [tx, tx+arr_len*cos_t], [ty, ty+arr_len*sin_t])

        # ── Wumpus ────────────────────────────────────────────────────────────
        wx,wy = wumpus.xy()
        self._wumpus_dot.set_data([wx],[wy])
        wump_col = '#ff4444' if wumpus.caught else _C_WMP
        self._wumpus_dot.set_markerfacecolor(wump_col)

        # ── Truck planned path ────────────────────────────────────────────────
        if truck._path and truck._pi < len(truck._path):
            remaining = truck._path[truck._pi:]
            px = [tx] + [p[0] for p in remaining[:60]]
            py = [ty] + [p[1] for p in remaining[:60]]
            self._tpath.set_data(px, py)
        else:
            self._tpath.set_data([], [])

        # ── Wumpus path ───────────────────────────────────────────────────────
        if wumpus.vis_path and wumpus._pi < len(wumpus.vis_path):
            cells = wumpus.vis_path[wumpus._pi:]
            wpx = [obs.cell_xy(*c)[0] for c in cells[:30]]
            wpy = [obs.cell_xy(*c)[1] for c in cells[:30]]
            self._wpath.set_data(wpx, wpy)
        else:
            self._wpath.set_data([], [])

        # ── Extinguish ring ───────────────────────────────────────────────────
        if truck.ext_target and truck.ext_progress > 0:
            ex,ey = truck.ext_target
            self._ext_ring.set_center((ex,ey))
            self._ext_ring.set_alpha(0.6)
            # Wedge as progress indicator
            angle_end = 90 - truck.ext_progress * 360
            self._ext_arc.set_center((ex,ey))
            self._ext_arc.set_radius(EXTINGUISH_R)
            self._ext_arc.theta1 = angle_end
            self._ext_arc.theta2 = 90
            self._ext_arc.set_alpha(0.25)
        else:
            self._ext_ring.set_alpha(0)
            self._ext_arc.set_alpha(0)

        # ── Fire spread preview (show radius of most urgent fire) ─────────────
        burning = obs.by_state(State.BURNING)
        if burning:
            # Most urgent = lowest time_to_spread
            urgent = min(burning, key=lambda c: obs.time_to_spread(*c, sim_time))
            t_left = obs.time_to_spread(*urgent, sim_time)
            if t_left < 4.0:
                bx,by = obs.cell_xy(*urgent)
                self._spread_ring.set_center((bx,by))
                self._spread_ring.set_radius(30.0)
                self._spread_ring.set_alpha(min(0.7, (4.0-t_left)/4.0))
            else:
                self._spread_ring.set_alpha(0)
        else:
            self._spread_ring.set_alpha(0)

        # ── Field time label ──────────────────────────────────────────────────
        self._field_t.set_text(f't = {sim_time:.0f} s')

        # ── Stats panel ───────────────────────────────────────────────────────
        self._t_time.set_text(f'Time:   {sim_time:6.0f} / {SIM_DURATION:.0f} s')
        mode_str = 'HUNT MODE  (catching Wumpus!)' if truck.hunt_mode else 'firefighting'
        mode_col  = '#ff4444' if truck.hunt_mode else '#44ff88'
        self._t_mode.set_text(f'Mode:  {mode_str}')
        self._t_mode.set_color(mode_col)

        self._t_ws.set_text(f'{wumpus.score} pts')
        self._t_ts.set_text(f'{truck.score} pts')

        n_int  = len(obs.by_state(State.INTACT))
        n_burn = len(obs.by_state(State.BURNING))
        n_ext  = len(obs.by_state(State.EXTINGUISHED))
        n_bur  = len(obs.by_state(State.BURNED))
        self._t_int.set_text(  f'Intact:       {n_int}')
        self._t_burn.set_text( f'Burning:      {n_burn}')
        self._t_ext.set_text(  f'Extinguished: {n_ext}')
        self._t_burned.set_text(f'Burned:       {n_bur}')

        self._t_wcpu.set_text(
            f'  Wumpus A*:  {wumpus.plan_time:.3f} s')
        self._t_tcpu.set_text(
            f'  Truck PRM:  {truck.plan_time + truck.prm.build_time:.3f} s  '
            f'(build {truck.prm.build_time:.2f}s)')

        pct = min(1.0, sim_time / SIM_DURATION)
        self._prog_bar.set_width(0.92 * pct)
        self._t_pct.set_text(f'{pct*100:.0f}%')

        self.fig.canvas.draw()
        plt.pause(VIZ_PAUSE)

    def wait_for_close(self):
        self.fig.suptitle(
            self.fig.texts[0].get_text() + '   [Close window to continue]',
            color='#ff8800', fontsize=15, fontweight='bold', y=0.98)
        self.fig.canvas.draw()
        plt.ioff()
        plt.show(block=True)


# ─────────────────────────────────────────────────────────────────────────────
# Single simulation run
# ─────────────────────────────────────────────────────────────────────────────

def _find_clear(grid, pr, pc):
    for dr in range(15):
        for dc in range(15):
            r=max(0,min(GRID_N-1,pr+dr)); c=max(0,min(GRID_N-1,pc+dc))
            if not grid[r,c]: return (r,c)
    return (pr,pc)


def run_simulation(seed, run_idx, record=False):
    grid, _ = generate_grid(seed)
    obs     = ObstacleManager(grid)

    # Start Wumpus in lower-left quadrant, truck in upper-right — not extreme corners
    # so fires can realistically be reached
    wumpus_cell = _find_clear(grid, 8,  8)
    truck_cell  = _find_clear(grid, GRID_N-16, GRID_N-16)
    tx0,ty0     = obs.cell_xy(*truck_cell)

    print(f'  Building PRM ({PRM_SAMPLES} nodes)...', end='', flush=True)
    prm = PRMPlanner(obs)
    prm.build(seed=seed)
    print(f' {prm.build_time:.2f}s', flush=True)

    wumpus = Wumpus(obs, wumpus_cell)
    truck  = Firetruck(obs, prm, (tx0, ty0, 0.0))

    n_obs   = len(obs.state)
    t       = 0.0
    frames  = []
    last_viz= -VIZ_INTERVAL
    wall_t0 = time.perf_counter()
    hunt_activated = False

    disp = LiveDisplay(run_idx, seed, n_obs) if _SHOW else None

    while t < SIM_DURATION:
        wumpus.step(t, DT)
        burned, _ = obs.update(t)
        wumpus.award_burn_points(len(burned))
        events = truck.step(DT, t)

        # Steady-state early exit: no intact/burning cells and Wumpus caught (or hunt done)
        if not obs.by_state(State.BURNING) and not obs.by_state(State.INTACT):
            if not hunt_activated:
                truck.activate_hunt(wumpus)
                hunt_activated = True
            # End as soon as Wumpus is caught
            if wumpus.caught:
                if disp:
                    disp.update(obs, wumpus, truck, t)
                break

        if disp and (t - last_viz) >= VIZ_INTERVAL:
            disp.update(obs, wumpus, truck, t)
            last_viz = t

        if record and int(t/DT) % max(1,int(SNAP_EVERY/DT)) == 0:
            sg = np.zeros((GRID_N,GRID_N),dtype=np.int8)
            for (r,c),s in obs.state.items(): sg[r,c]=s.value
            frames.append({'grid':sg.copy(),'wumpus':wumpus.cell,
                           'truck':truck.xy(),'t':t,
                           'ws':wumpus.score,'ts':truck.score})

        t += DT

    if disp:
        disp.update(obs, wumpus, truck, t)
        disp.wait_for_close()

    n_ext   = len(obs.by_state(State.EXTINGUISHED))
    ext_pct = 100*n_ext/max(1,n_obs)

    return {
        'run_seed':        seed,
        'sim_time':        t,
        'wall_time':       time.perf_counter()-wall_t0,
        'wumpus_score':    wumpus.score,
        'truck_score':     truck.score,
        'wumpus_plan_t':   wumpus.plan_time,
        'truck_plan_t':    truck.plan_time,
        'prm_build_t':     prm.build_time,
        'truck_total_cpu': truck.plan_time+prm.build_time,
        'n_obstacles':     n_obs,
        'n_extinguished':  n_ext,
        'ext_pct':         ext_pct,
        'wumpus_caught':   wumpus.caught,
    }, frames, obs, grid, wumpus, truck


# ─────────────────────────────────────────────────────────────────────────────
# Output plots
# ─────────────────────────────────────────────────────────────────────────────

_COL_HEX = {
    State.INTACT:       '#2d6a2d',
    State.BURNING:      '#e8450a',
    State.EXTINGUISHED: '#2176ae',
    State.BURNED:       '#3a3a3a',
}
_BG_PLT = '#d6cfc0'

def _save_score_table(results, out):
    fig,ax=plt.subplots(figsize=(12,3.5)); ax.axis('off')
    ww=tw=0; rows=[]
    for r in results:
        w,t=r['wumpus_score'],r['truck_score']
        win='Wumpus' if w>t else ('Truck' if t>w else 'Draw')
        if win=='Wumpus': ww+=1
        if win=='Truck':  tw+=1
        caught = 'Yes' if r.get('wumpus_caught') else 'No'
        rows.append([r['run_seed'],w,t,win,
                     f"{r['n_extinguished']}/{r['n_obstacles']} ({r['ext_pct']:.0f}%)",
                     caught, f"{r['sim_time']:.0f}s"])
    champ='WUMPUS' if ww>=3 else 'TRUCK' if tw>=3 else 'No champion'
    rows.append(['TOTAL',
                 sum(r['wumpus_score'] for r in results),
                 sum(r['truck_score']  for r in results),
                 f'Champion: {champ}','-','-','-'])
    tbl=ax.table(
        cellText=rows,
        colLabels=['Seed','Wumpus','Truck','Winner',
                   'Extinguished','Wumpus caught','Sim time'],
        loc='center',cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1,1.7)
    for j in range(7):
        tbl[0,j].set_facecolor('#1a1a2e')
        tbl[0,j].set_text_props(color='white',fontweight='bold')
    for i,row in enumerate(rows,1):
        if 'Wumpus' in str(row[3]): tbl[i,3].set_facecolor('#fce0d5')
        elif 'Truck' in str(row[3]): tbl[i,3].set_facecolor('#d5e8fc')
    ax.set_title('WILDFIRE — Simulation results (5 runs)',
                 fontsize=13,fontweight='bold',pad=10)
    plt.tight_layout(); plt.savefig(out,dpi=130,bbox_inches='tight'); plt.close()
    print(f'  Scores      -> {out}')

def _save_cpu_chart(results, out):
    fig,ax=plt.subplots(figsize=(10,5))
    labels=[f"Run {r['run_seed']}" for r in results]
    wt=[r['wumpus_plan_t'] for r in results]
    pb=[r['prm_build_t']   for r in results]
    pq=[r['truck_plan_t']  for r in results]
    x,w=np.arange(len(labels)),0.32
    ax.bar(x-w/2,wt,w,label='Wumpus A* (planning)',color='#9b00cc',alpha=0.88)
    ax.bar(x+w/2,pb,w,label='Truck PRM build',     color='#1060A0',alpha=0.88)
    ax.bar(x+w/2,pq,w,bottom=pb,label='Truck PRM query',color='#60A0E0',alpha=0.88)
    ax.set_xlabel('Run (seed)',fontsize=11); ax.set_ylabel('CPU time (s)',fontsize=11)
    ax.set_title('Planner CPU time — A* (Wumpus) vs PRM (Firetruck)',fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    tw=sum(wt); tt=sum(r['truck_total_cpu'] for r in results)
    ax.text(0.99,0.95,f'Total A*: {tw:.2f}s  |  Total PRM: {tt:.2f}s',
            transform=ax.transAxes,ha='right',va='top',fontsize=9,color='#444')
    plt.tight_layout(); plt.savefig(out,dpi=130); plt.close()
    print(f'  CPU chart   -> {out}')

def _save_env(obs, grid, wumpus, truck, seed, t_end, out):
    fig,ax=plt.subplots(figsize=(8,8))
    ax.set_facecolor('#111'); ax.set_xlim(0,FIELD_SIZE); ax.set_ylim(0,FIELD_SIZE)
    ax.set_aspect('equal')
    ax.set_title(f'Final state  seed={seed}  t={t_end:.0f}s',fontsize=11,color='white')
    fig.patch.set_facecolor('#111')
    for (r,c),s in obs.state.items():
        ax.add_patch(mpatches.Rectangle(
            (c*CELL_SIZE,r*CELL_SIZE),CELL_SIZE,CELL_SIZE,
            linewidth=0,facecolor=_COL_HEX[s],alpha=0.92))
    wx,wy=wumpus.xy(); tx,ty=truck.xy()
    ax.plot(wx,wy,'s',ms=12,color=_C_WMP,  zorder=6)
    ax.plot(tx,ty,'^',ms=12,color=_C_TRUCK,zorder=6)
    ax.legend(handles=[
        mpatches.Patch(color=_COL_HEX[State.INTACT],      label='Intact'),
        mpatches.Patch(color=_COL_HEX[State.BURNING],     label='Burning'),
        mpatches.Patch(color=_COL_HEX[State.EXTINGUISHED],label='Extinguished'),
        mpatches.Patch(color=_COL_HEX[State.BURNED],      label='Burned'),
    ],fontsize=8,loc='upper right',facecolor='#222',labelcolor='white')
    plt.tight_layout(); plt.savefig(out,dpi=130); plt.close()
    print(f'  Env snap    -> {out}')

def _save_anim(frames, obs_ref, grid, out):
    if not frames: print('  No frames — skipping animation.'); return
    snaps=frames[::max(1,len(frames)//80)]
    fig,ax=plt.subplots(figsize=(7,7)); fig.patch.set_facecolor('#111')

    def _draw(i):
        ax.clear(); snap=snaps[i]; g=snap['grid']
        ax.set_facecolor('#111'); ax.set_xlim(0,FIELD_SIZE); ax.set_ylim(0,FIELD_SIZE)
        ax.set_aspect('equal')
        ax.set_title(f'WILDFIRE  t={snap["t"]:.0f}s  '
                     f'Wumpus:{snap["ws"]}  Truck:{snap["ts"]}',
                     fontsize=10,color='white')
        for r in range(GRID_N):
            for c in range(GRID_N):
                if not grid[r,c]: continue
                ax.add_patch(mpatches.Rectangle(
                    (c*CELL_SIZE,r*CELL_SIZE),CELL_SIZE,CELL_SIZE,
                    linewidth=0,facecolor=_COL_HEX[State(int(g[r,c]))],alpha=0.92))
        wr,wc=snap['wumpus']
        wx,wy=obs_ref.cell_xy(wr,wc)
        ax.plot(wx,wy,'s',ms=10,color=_C_WMP,  zorder=6)
        ax.plot(*snap['truck'],'^',ms=10,color=_C_TRUCK,zorder=6)

    anim=animation.FuncAnimation(fig,_draw,frames=len(snaps),interval=int(1000/ANIM_FPS))
    anim.save(out,writer='pillow',fps=ANIM_FPS); plt.close()
    print(f'  Animation   -> {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

from config import PRM_SAMPLES

def main():
    print("""
+================================================================+
|   RBE-550 Motion Planning  --  Assignment 5: WILDFIRE          |
|   Samuel Oluwakorede Oyefusi  |  WPI  |  Spring 2026           |
+================================================================+
""")
    if not _SHOW:
        print('  [headless mode — no popup window]\n')

    results=[]; rec=None

    for i,seed in enumerate(SEEDS):
        record=(i==0)
        print(f'\nRun {i+1}/{N_RUNS}  seed={seed}')
        result,frames,obs,grid,wumpus,truck=run_simulation(
            seed=seed,run_idx=i,record=record)
        result['run_seed']=seed
        results.append(result)
        if record: rec=(frames,obs,grid,wumpus,truck,result)

        caught_str = '  [Wumpus caught!]' if result['wumpus_caught'] else ''
        print(f'  Wumpus={result["wumpus_score"]} pts  '
              f'Truck={result["truck_score"]} pts  '
              f'Extinguished={result["n_extinguished"]}/{result["n_obstacles"]} '
              f'({result["ext_pct"]:.0f}%)  '
              f'wall={result["wall_time"]:.1f}s{caught_str}')

    # ── Console table ─────────────────────────────────────────────────────────
    print(f'\n{"Seed":>6} {"Wumpus":>8} {"Truck":>7} {"Winner":<10} '
          f'{"Ext%":>6} {"Caught":>7} {"SimT":>7}')
    print('─'*62)
    ww=tw=0
    for r in results:
        w,t=r['wumpus_score'],r['truck_score']
        win='Wumpus' if w>t else ('Truck' if t>w else 'Draw')
        if win=='Wumpus': ww+=1
        if win=='Truck':  tw+=1
        caught='Yes' if r.get('wumpus_caught') else 'No'
        print(f'{r["run_seed"]:>6} {w:>8} {t:>7} {win:<10} '
              f'{r["ext_pct"]:>5.0f}% {caught:>7} {r["sim_time"]:>6.0f}s')
    print('─'*62)
    champ='WUMPUS' if ww>=3 else 'TRUCK' if tw>=3 else 'No champion'
    print(f'Champion (best of 5): {champ}\n')

    # ── Outputs ───────────────────────────────────────────────────────────────
    print('Saving outputs...')
    _save_score_table(results, os.path.join(_DIR,'wildfire_scores.png'))
    _save_cpu_chart(results,   os.path.join(_DIR,'wildfire_cpu.png'))
    if rec:
        frames,obs,grid,wumpus,truck,result=rec
        _save_env(obs,grid,wumpus,truck,result['run_seed'],result['sim_time'],
                  os.path.join(_DIR,'wildfire_env.png'))
        _save_anim(frames,obs,grid,os.path.join(_DIR,'wildfire_anim.gif'))
    print(f'\nDone. Files saved to: {_DIR}')


if __name__=='__main__':
    main()
