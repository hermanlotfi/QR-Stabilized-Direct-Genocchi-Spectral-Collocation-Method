#hhh
import numpy as np
import matplotlib.pyplot as plt
from math import comb
from pathlib import Path
from scipy.special import gamma
from scipy.linalg import solve, solve_triangular
import pandas as pd
import time

# ================================================================
# QR-STABILIZED DIRECT GENOCCHI-RIESZ SOLVER
# Four benchmark examples
# ================================================================
# Example 1: 1D manufactured nonlinear reaction-diffusion
# Example 2: 2D linear Riesz diffusion benchmark (Zhu et al.)
# Example 3: 2D Fisher-type Riesz reaction-diffusion benchmark (Zhu et al.)
# Example 4: 2D Allen-Cahn benchmark (Almushaira)
# ================================================================

OUT = Path('genocchi_q1_cpu_results')
OUT.mkdir(exist_ok=True)
FIG = OUT/'figures'
FIG.mkdir(exist_ok=True)

ALPHAS_E1 = [1.2, 1.5, 1.8]
N_SPATIAL_E1 = [4, 5, 6, 8, 10, 12, 14]
DT_SPATIAL_E1 = 0.005
N_TEMP_E1 = 12
DT_TEMP_E1 = [0.02, 0.01, 0.005, 0.0025]

E2_ALPHAS = [1.6, 1.8]
E2_N_SPATIAL = [5, 6, 8, 10, 12]
E2_DT_SPATIAL = 0.01
E2_N_TEMP = 10
E2_DT_TEMP = [0.02, 0.01, 0.005, 0.0025]

E3_ALPHAS = [1.7, 1.9]
E3_N_SPATIAL = [5, 6, 8, 10, 12]
E3_DT_SPATIAL = 0.01
E3_N_TEMP = 10
E3_DT_TEMP = [0.02, 0.01, 0.005, 0.0025]

E4_ALPHAS = [1.3, 1.7]
E4_N_SPATIAL = [8, 9, 10, 11, 12, 14]
E4_DT_SPATIAL = 0.01
E4_N_TEMP = 12
E4_DT_TEMP = [0.25, 0.125, 0.0625, 0.03125]

T_SHORT = 0.10
T_E2E3 = 1.0
T_E4 = 1.0
KAPPA1 = 1.0

NEWTON_TOL = 1e-12
NEWTON_MAXIT = 25
LINESEARCH_MAXIT = 20

# ================================================================
# Genocchi
# ================================================================

def genocchi_numbers(N):
    G = np.zeros(N+1)
    if N >= 1:
        G[1] = 1.0
    for n in range(2, N+1):
        G[n] = -0.5*sum(comb(n,k)*G[k] for k in range(n))
    return G


def genocchi_coeff_matrix(N):
    G = genocchi_numbers(N)
    C = np.zeros((N,N))
    for j in range(1,N+1):
        for k in range(j):
            C[k,j-1] = comb(j,k)*G[j-k]
    return C


def cgl_points(N):
    j = np.arange(N)
    z = np.cos(np.pi*j/(N-1))
    return (1-z)/2


def eval_basis(x,C):
    x=np.asarray(x,float); N=C.shape[1]
    P=np.empty((len(x),N))
    for j in range(N):
        P[:,j]=np.polynomial.polynomial.polyval(x,C[:,j])
    return P

# ================================================================
# Fractional operators
# ================================================================

def left_rl_monomial(x,m,a):
    x=np.asarray(x,float); out=np.zeros_like(x); mask=x>0
    out[mask]=gamma(m+1)/gamma(m+1-a)*x[mask]**(m-a)
    return out


def right_rl_monomial(x,m,a):
    x=np.asarray(x,float); y=1-x; out=np.zeros_like(x); mask=y>0
    for k in range(m+1):
        out[mask] += ((-1.)**k*comb(m,k)*gamma(k+1)/gamma(k+1-a)*y[mask]**(k-a))
    return out


def riesz_monomial(x,m,a):
    fac=-1/(2*np.cos(a*np.pi/2))
    return fac*(left_rl_monomial(x,m,a)+right_rl_monomial(x,m,a))


def direct_riesz(x,C,a):
    R=np.zeros((len(x),C.shape[1]))
    for j in range(C.shape[1]):
        for m in range(C.shape[0]):
            if C[m,j] != 0:
                R[:,j]+=C[m,j]*riesz_monomial(x,m,a)
    return R

# ================================================================
# Polynomial coefficient utilities
# ================================================================

def polynomial_monomial_coeffs_from_product_power(power):
    # x^p(1-x)^p = sum_{j=0}^p (-1)^j C(p,j)x^{p+j}
    d=2*power
    a=np.zeros(d+1)
    for j in range(power+1):
        a[power+j]=(-1.)**j*comb(power,j)
    return a


def genocchi_coeffs_for_polynomial(monomial_coeffs,N):
    d=len(monomial_coeffs)-1
    if N<d+1: raise ValueError('N too small')
    C=genocchi_coeff_matrix(N)
    return solve(C[:d+1,:d+1],monomial_coeffs)

def cgl_barycentric_weights(N):
    w = (-1.0) ** np.arange(N, dtype=float)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def barycentric_interp_matrix(x_nodes, x_eval):
    x_nodes=np.asarray(x_nodes,dtype=float)
    x_eval=np.asarray(x_eval,dtype=float)
    w=cgl_barycentric_weights(len(x_nodes))
    I=np.empty((len(x_eval),len(x_nodes)),dtype=float)
    for r,xx in enumerate(x_eval):
        d=xx-x_nodes
        j=np.argmin(np.abs(d))
        if abs(d[j])<5e-14:
            I[r,:]=0.0
            I[r,j]=1.0
        else:
            q=w/d
            I[r,:]=q/np.sum(q)
    return I


# ================================================================
# Generic 1D stable setup
# ================================================================

def setup_1d(N,alpha):
    C=genocchi_coeff_matrix(N); x=cgl_points(N); Phi=eval_basis(x,C)
    Q,RG=np.linalg.qr(Phi,mode='reduced')
    idx=np.arange(1,N-1); xi=x[idx]
    Rint=direct_riesz(xi,C,alpha)
    B=solve_triangular(RG.T,Rint.T,lower=True).T
    return {'N':N,'a':alpha,'C':C,'x':x,'Phi':Phi,'Q':Q,'RG':RG,'idx':idx,'xi':xi,'Rint':Rint,'B':B}


def cn_step_1d(setup,c_old,c_init,dt,t0,t1,reaction,reaction_prime,source,kappa):
    x=setup['x']; Q=setup['Q']; B=setup['B']; idx=setup['idx']; a=setup['a']
    Qi=Q[idx,:]; Qb=Q[[0,-1],:]; xi=x[idx]
    y_old=setup.get('y_old_current',None)
    y=c_init.copy()
    f0=source(xi,t0,a,kappa); f1=source(xi,t1,a,kappa)
    uold=Qi@c_old
    for it in range(NEWTON_MAXIT):
        unew=Qi@y
        Fint=(unew-uold)/dt -0.5*kappa*(B@y+B@c_old)+0.5*(reaction(unew)+reaction(uold))-0.5*(f1+f0)
        F=np.r_[Fint,Qb@y]
        fn=np.linalg.norm(F,np.inf)
        if fn<NEWTON_TOL: return y,it+1,fn
        Jint=Qi/dt-0.5*kappa*B+0.5*reaction_prime(unew)[:,None]*Qi
        J=np.vstack([Jint,Qb])
        delta=solve(J,-F)
        lam=1.
        ok=False
        for _ in range(LINESEARCH_MAXIT):
            yt=y+lam*delta
            ut=Qi@yt
            Ftint=(ut-uold)/dt-0.5*kappa*(B@yt+B@c_old)+0.5*(reaction(ut)+reaction(uold))-0.5*(f1+f0)
            Ft=np.r_[Ftint,Qb@yt]
            if np.linalg.norm(Ft,np.inf) < fn:
                ok=True; y=yt; break
            lam*=0.5
        if not ok:
            y=y+lam*delta
    raise RuntimeError(f'1D Newton failed at t={t1}')

# ================================================================
# Example 1
# ================================================================

def e1_p(x): return x**2*(1-x)**2

def e1_Rp(x,a): return riesz_monomial(x,2,a)-2*riesz_monomial(x,3,a)+riesz_monomial(x,4,a)

def e1_reaction(u): return u*(1-u)

def e1_reaction_prime(u): return 1-2*u

def e1_source(x,t,a,k): return -k*np.exp(-t)*e1_Rp(x,a)-np.exp(-2*t)*e1_p(x)**2

def run_e1(alpha,N,dt,T):
    cpu_start = time.perf_counter()
    s=setup_1d(N,alpha)
    # Interpolatory initial representation in stable QR coordinates.
    # It is exact whenever the polynomial degree is representable.
    u0=exact_solution_1d=s['x']**2*(1.0-s['x'])**2
    y=s['Q'].T@u0
    nsteps=int(round(T/dt))
    for n in range(nsteps):
        t0=n*dt; t1=(n+1)*dt
        y,nit,fn=cn_step_1d(s,y,y,dt,t0,t1,e1_reaction,e1_reaction_prime,e1_source,KAPPA1)
    xd=np.linspace(0,1,1001); I=barycentric_interp_matrix(s['x'],xd)
    u_nodes=s['Q']@y
    un=I@u_nodes; ue=np.exp(-T)*e1_p(xd); err=np.abs(un-ue)
    cpu_time = time.perf_counter() - cpu_start
    return {'alpha':alpha,'N':N,'dt':dt,'T':T,'Linf':err.max(),'L2':np.sqrt(np.trapezoid(err**2,xd)),'CPU_s':cpu_time,'y':y,'setup':s,'nit':nit}

# ================================================================
# Generic 2D stable tensor setup
# ================================================================

def setup_2d(N,ax,ay=None):
    if ay is None: ay=ax
    Cx=genocchi_coeff_matrix(N); Cy=genocchi_coeff_matrix(N)
    x=cgl_points(N); y=x.copy();
    Phix=eval_basis(x,Cx); Phiy=eval_basis(y,Cy)
    Qx,Rx=np.linalg.qr(Phix,mode='reduced'); Qy,Ry=np.linalg.qr(Phiy,mode='reduced')
    ii=np.arange(1,N-1)
    xi=x[ii]; yi=y[ii]
    Rxint=direct_riesz(xi,Cx,ax); Ryint=direct_riesz(yi,Cy,ay)
    Bx=solve_triangular(Rx.T,Rxint.T,lower=True).T
    By=solve_triangular(Ry.T,Ryint.T,lower=True).T
    Qxi=Qx[ii,:]; Qyi=Qy[ii,:]
    Lx=np.kron(Qyi,Bx); Ly=np.kron(By,Qxi)
    L=Lx+Ly
    Q2=np.kron(Qy,Qx)
    mask_full=[]; mask_int=[]
    for j in range(N):
        for i in range(N):
            k=i+j*N
            if i==0 or i==N-1 or j==0 or j==N-1: mask_full.append(k)
            else: mask_int.append((i-1)+(N-2)*(j-1))
    ib=[]
    for j in range(N):
        for i in range(N):
            if i==0 or i==N-1 or j==0 or j==N-1:
                ib.append(i+j*N)
    return {'N':N,'ax':ax,'ay':ay,'Cx':Cx,'Cy':Cy,'x':x,'y':y,'Phix':Phix,'Phiy':Phiy,'Qx':Qx,'Qy':Qy,'Rx':Rx,'Ry':Ry,'ii':ii,'xi':xi,'yi':yi,'Qxi':Qxi,'Qyi':Qyi,'Bx':Bx,'By':By,'L':L,'Q2':Q2,'Q2_int':np.kron(Qyi,Qxi),'boundary_full_idx':np.array(ib,dtype=int),'interior_grid_shape':(N-2,N-2)}


def e2_spatial(N):
    return polynomial_monomial_coeffs_from_product_power(2)

def e4_spatial(N):
    return polynomial_monomial_coeffs_from_product_power(5)


def tensor_exact_Y(setup,cx,cy,scale=1.0):
    return scale*(setup['Rx']@cx[:,None]@ (setup['Ry']@cy[:,None]).T)


def tensor_y_from_coeff_matrix(setup,Ccoef):
    Y=setup['Rx']@Ccoef@setup['Ry'].T
    return Y.reshape(-1,order='F')


def eval_2d_dense(setup,yvec,npts=161):
    N=setup['N']
    xd=np.linspace(0,1,npts)
    yd=xd.copy()
    Ix=barycentric_interp_matrix(setup['x'],xd)
    Iy=barycentric_interp_matrix(setup['y'],yd)
    Ycoef=yvec.reshape((N,N),order='F')
    U_nodes=(setup['Qx']@Ycoef@setup['Qy'].T)
    U_xy=Ix@U_nodes@Iy.T
    X,Ygrid=np.meshgrid(xd,yd,indexing='xy')
    return X,Ygrid,U_xy.T


def build_2d_residual_jac(setup,yold,ynew,dt,t0,t1,reaction,reaction_prime,source,kx,ky):
    N=setup['N']; nint=(N-2)**2; Q2=setup['Q2']; L=kx*setup['Lx'] if 'Lx' in setup else None
    # setup['L'] has unit coefficients; rebuild anisotropic if needed
    Ldiff=kx*np.kron(setup['Qyi'],setup['Bx']) + ky*np.kron(setup['By'],setup['Qxi'])
    bidx=setup['boundary_full_idx']; ii=setup['ii']; Qint=np.kron(setup['Qyi'],setup['Qxi']); Qb=Q2[bidx,:]
    xi=setup['xi']; yi=setup['yi']; X=xi[:,None]; Y=yi[None,:]
    uold=(Qint@yold); unew=(Qint@ynew)
    # source expects x,y arrays
    f0=source(X,Y,t0); f1=source(X,Y,t1)
    f0=f0.reshape(-1,order='F'); f1=f1.reshape(-1,order='F')
    Fint=(unew-uold)/dt-0.5*(Ldiff@ynew+Ldiff@yold)+0.5*(reaction(unew)+reaction(uold))-0.5*(f1+f0)
    F=np.r_[Fint,Qb@ynew]
    Jint=Qint/dt-0.5*Ldiff+0.5*reaction_prime(unew)[:,None]*Qint
    J=np.vstack([Jint,Qb])
    return F,J

# ================================================================
# Example 2 and 3 source functions
# ================================================================

def p2(x): return x**2*(1-x)**2

def rp2(x,a): return riesz_monomial(x,2,a)-2*riesz_monomial(x,3,a)+riesz_monomial(x,4,a)

def e2_source_factory(ax,ay,kx=1.0,ky=1.0):
    def src(x,y,t):
        return -100*np.exp(-t)*(p2(x)*p2(y)+kx*rp2(x,ax)*p2(y)+ky*p2(x)*rp2(y,ay))
    return src

def e3_source_factory(ax,ay,kx=0.5,ky=0.75):
    def src(x,y,t):
        u=100*np.exp(-t)*p2(x)*p2(y)
        diff=100*np.exp(-t)*(kx*rp2(x,ax)*p2(y)+ky*p2(x)*rp2(y,ay))
        return -100*np.exp(-t)*p2(x)*p2(y)-diff-0.1*u*(1-u)
    return src

def run_2d_example(N,dt,T,ax,ay,kx,ky,reaction,reaction_prime,source,cx,cy,scale,exact_fun,name):
    cpu_start = time.perf_counter()
    s=setup_2d(N,ax,ay);
    # Stable nodal interpolation/projection of the exact initial data.
    U0=exact_fun(s['x'][:,None],s['y'][None,:],0.0)
    y=s['Q2'].T@U0.reshape(-1,order='F')
    nsteps=int(round(T/dt))
    for n in range(nsteps):
        t0=n*dt; t1=(n+1)*dt
        yn=y.copy()
        for it in range(NEWTON_MAXIT):
            F,J=build_2d_residual_jac(s,y,yn,dt,t0,t1,reaction,reaction_prime,source,kx,ky)
            fn=np.linalg.norm(F,np.inf)
            if fn<NEWTON_TOL: break
            delta=solve(J,-F); lam=1.; ok=False
            for _ in range(LINESEARCH_MAXIT):
                yt=yn+lam*delta
                Ft,_=build_2d_residual_jac(s,y,yt,dt,t0,t1,reaction,reaction_prime,source,kx,ky)
                if np.linalg.norm(Ft,np.inf)<fn: yn=yt; ok=True; break
                lam*=0.5
            if not ok: raise RuntimeError(f'{name}: line search failed at t={t1}, it={it+1}')
        else:
            raise RuntimeError(f'{name}: Newton failed at t={t1}')
        y=yn
    X,Y,U=eval_2d_dense(s,y,161)
    UE=exact_fun(X,Y,T)
    err=np.abs(U-UE)
    cpu_time = time.perf_counter() - cpu_start
    return {'alpha':ax,'alpha_y':ay,'N':N,'dt':dt,'T':T,'Linf':err.max(),'L2':np.sqrt(np.trapezoid(np.trapezoid(err**2, X[0,:],axis=1),Y[:,0])),'CPU_s':cpu_time,'y':y,'setup':s,'U':U,'UE':UE,'X':X,'Y':Y}



# ================================================================
# LaTeX / report generation
# ================================================================

def fmt_sci(x, digits=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '--'
    return f"{x:.{digits}e}"


def latex_escape(value):
    text = str(value)
    return (text.replace('_', r'\\_')
                .replace('%', r'\\%')
                .replace('&', r'\\&'))


def write_latex_table(df, columns, headers, filename,
                      caption, label, align=None,
                      scientific_cols=None, order_cols=None, decimal_cols=None):
    scientific_cols = scientific_cols or set()
    order_cols = order_cols or set()
    decimal_cols = decimal_cols or set()
    if align is None:
        align = 'c' * len(columns)

    lines = []
    lines.append('\\begin{table}[H]')
    lines.append('\\centering')
    lines.append('\\caption{' + caption + '}')
    lines.append('\\label{' + label + '}')
    lines.append('\\begin{tabular}{' + align + '}')
    lines.append('\\toprule')
    lines.append(' & '.join(headers) + ' \\\\')
    lines.append('\\midrule')

    for _, row in df.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if col in scientific_cols:
                vals.append(fmt_sci(float(value)))
            elif col in order_cols:
                vals.append('--' if pd.isna(value) else f'{float(value):.4f}')
            elif col in decimal_cols:
                vals.append(f'{float(value):.4f}')
            elif isinstance(value, (float, np.floating)):
                vals.append(f'{float(value):.6g}')
            else:
                vals.append(str(value))
        lines.append(' & '.join(vals) + ' \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    path = OUT / filename
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


def generate_latex_tables():
    table_dir = OUT / 'tables'
    table_dir.mkdir(exist_ok=True)

    e1s = pd.read_csv(OUT/'example1_spatial.csv')
    e1t = pd.read_csv(OUT/'example1_temporal.csv')
    e2s = pd.read_csv(OUT/'example2_spatial.csv')
    e2t = pd.read_csv(OUT/'example2_temporal.csv')
    e3s = pd.read_csv(OUT/'example3_spatial.csv')
    e3t = pd.read_csv(OUT/'example3_temporal.csv')
    e4s = pd.read_csv(OUT/'example4_spatial.csv')
    e4t = pd.read_csv(OUT/'example4_temporal.csv')
    cond = pd.read_csv(OUT/'conditioning.csv')

    # Copy each generated table to table_dir
    specs = [
        (e1s, ['N','alpha','Linf','L2','CPU_s'],
         [r'$N$',r'$\alpha$',r'$E_\infty$',r'$E_2$',r'CPU (s)'],
         'example1_spatial_table.tex',
         r'Example 1: spatial errors for $T=0.1$ and $\Delta t=0.005$.',
         'tab:e1spatial', 'crrrr', {'Linf','L2'}, set(), {'CPU_s'}),
        (e1t, ['alpha','dt','Linf','order','CPU_s'],
         [r'$\alpha$',r'$\Delta t$',r'$E_\infty$',r'Order',r'CPU (s)'],
         'example1_temporal_table.tex',
         r'Example 1: temporal convergence for $N=12$ and $T=0.1$.',
         'tab:e1temporal', 'crrrr', {'Linf'}, {'order'}, {'CPU_s'}),
        (e2s, ['alpha','N','Linf','L2','CPU_s'],
         [r'$\alpha$',r'$N$',r'$E_\infty$',r'$E_2$',r'CPU (s)'],
         'example2_spatial_table.tex',
         r'Example 2: spatial-resolution results at $T=1$ and $\Delta t=0.01$.',
         'tab:e2spatial', 'crrrr', {'Linf','L2'}, set(), {'CPU_s'}),
        (e2t, ['alpha','dt','Linf','order','CPU_s'],
         [r'$\alpha$',r'$\Delta t$',r'$E_\infty$',r'Order',r'CPU (s)'],
         'example2_temporal_table.tex',
         r'Example 2: temporal convergence for $N=10$ and $T=1$.',
         'tab:e2temporal', 'crrrr', {'Linf'}, {'order'}, {'CPU_s'}),
        (e3s, ['alpha','N','Linf','L2','CPU_s'],
         [r'$\alpha$',r'$N$',r'$E_\infty$',r'$E_2$',r'CPU (s)'],
         'example3_spatial_table.tex',
         r'Example 3: spatial-resolution results at $T=1$ and $\Delta t=0.01$.',
         'tab:e3spatial', 'crrrr', {'Linf','L2'}, set(), {'CPU_s'}),
        (e3t, ['alpha','dt','Linf','order','CPU_s'],
         [r'$\alpha$',r'$\Delta t$',r'$E_\infty$',r'Order',r'CPU (s)'],
         'example3_temporal_table.tex',
         r'Example 3: temporal convergence for $N=10$ and $T=1$.',
         'tab:e3temporal', 'crrrr', {'Linf'}, {'order'}, {'CPU_s'}),
        (e4s, ['alpha','N','Linf','L2','CPU_s'],
         [r'$\alpha$',r'$N$',r'$E_\infty$',r'$E_2$',r'CPU (s)'],
         'example4_spatial_table.tex',
         r'Example 4: Allen--Cahn spatial-resolution results at $T=1$ and $\Delta t=0.01$.',
         'tab:e4spatial', 'crrrr', {'Linf','L2'}, set(), {'CPU_s'}),
        (e4t, ['alpha','dt','Linf','CPU_s'],
         [r'$\alpha$',r'$\Delta t$',r'$E_\infty$',r'CPU (s)'],
         'example4_time_check_table.tex',
         r'Example 4: temporal exactness check for the polynomial-in-time manufactured solution.',
         'tab:e4timecheck', 'crrr', {'Linf'}, set(), {'CPU_s'}),
        (cond, ['alpha','N','cond_Phi','cond_Riesz','cond_A','cond_Jy','cond_B','CPU_s'],
         [r'$\alpha$',r'$N$',r'$\kappa_2(\Phi)$',r'$\kappa_2(R_I)$',r'$\kappa_2(A_c)$',r'$\kappa_2(J_y)$',r'$\kappa_2(B)$',r'CPU (s)'],
         'conditioning_table.tex',
         r'Conditioning study for the direct Genocchi and QR-stabilized systems.',
         'tab:conditioning', 'ccrrrrrr', {'cond_Phi','cond_Riesz','cond_A','cond_Jy','cond_B'}, set(), {'CPU_s'}),
    ]

    for spec in specs:
        path = write_latex_table(*spec)
        path.replace(table_dir / path.name)

    return table_dir


def save_representative_2d_plot(result, prefix, title, surface_label):
    X = result['X']
    Y = result['Y']
    U = result['U']
    UE = result['UE']
    err = np.abs(U-UE)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X,Y,U,rstride=4,cstride=4,linewidth=0,antialiased=True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.65, pad=0.10, label=surface_label)
    fig.tight_layout()
    fig.savefig(FIG/f'{prefix}_solution_surface.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(7,5.5))
    ax = fig.add_subplot(111)
    im = ax.contourf(X,Y,err,levels=40)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title + ' -- absolute error')
    fig.colorbar(im, ax=ax, label=r'$|e|$')
    fig.tight_layout()
    fig.savefig(FIG/f'{prefix}_absolute_error.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


# ================================================================
# Main studies
# ================================================================

def convergence_order(errors,steps):
    out=[np.nan]
    for i in range(1,len(errors)):
        out.append(np.log(errors[i-1]/errors[i])/np.log(steps[i-1]/steps[i]))
    return out


def run_e1_studies():
    rows=[]
    temporal=[]
    for a in ALPHAS_E1:
        spatial_errors=[]
        for N in N_SPATIAL_E1:
            r=run_e1(a,N,DT_SPATIAL_E1,T_SHORT)
            spatial_errors.append(r['Linf'])
            rows.append({'example':1,'alpha':a,'N':N,'dt':DT_SPATIAL_E1,'T':T_SHORT,'Linf':r['Linf'],'L2':r['L2'],'CPU_s':r['CPU_s']})
            if a == 1.5 and N == 12:
                xd = np.linspace(0.0, 1.0, 1001)
                I = barycentric_interp_matrix(r['setup']['x'], xd)
                un = I @ (r['setup']['Q'] @ r['y'])
                ue = np.exp(-T_SHORT) * e1_p(xd)
                ee = np.abs(un-ue)
                fig, ax = plt.subplots(figsize=(8,5))
                ax.plot(xd, ue, linewidth=2.0, label='Exact')
                ax.plot(xd, un, '--', linewidth=1.8, label='QR-stabilized Genocchi')
                ax.set_xlabel('x'); ax.set_ylabel('u(x,T)')
                ax.set_title(r'Example 1 solution ($\alpha=1.5$, $N=12$, $\Delta t=0.005$)')
                ax.grid(True, alpha=.3); ax.legend(); fig.tight_layout()
                fig.savefig(FIG/'example1_solution.png', dpi=300, bbox_inches='tight'); plt.close(fig)
                fig, ax = plt.subplots(figsize=(8,5))
                ax.semilogy(xd, np.maximum(ee,1e-18), linewidth=2.0)
                ax.set_xlabel('x'); ax.set_ylabel(r'$|e(x,T)|$')
                ax.set_title(r'Example 1 absolute error ($\alpha=1.5$, $N=12$)')
                ax.grid(True, which='both', alpha=.3); fig.tight_layout()
                fig.savefig(FIG/'example1_absolute_error.png', dpi=300, bbox_inches='tight'); plt.close(fig)
        # temporal
        errs=[]
        temp_runs=[]
        for dt in DT_TEMP_E1:
            r=run_e1(a,N_TEMP_E1,dt,T_SHORT)
            errs.append(r['Linf'])
            temp_runs.append(r)
        orders=convergence_order(errs,DT_TEMP_E1)
        for r,o in zip(temp_runs,orders):
            temporal.append({'example':1,'alpha':a,'N':N_TEMP_E1,'dt':r['dt'],'T':T_SHORT,'Linf':r['Linf'],'order':o,'CPU_s':r['CPU_s']})
        plt.semilogy(N_SPATIAL_E1,spatial_errors,'o-',label=rf'$\alpha={a}$')
    plt.xlabel('N'); plt.ylabel(r'$L^\infty$ error'); plt.grid(True,which='both',alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'example1_spatial.png',dpi=300); plt.close()
    plt.figure(figsize=(8,5))
    for a in ALPHAS_E1:
        z=[row['Linf'] for row in temporal if row['alpha']==a]
        plt.loglog(DT_TEMP_E1,z,'o-',label=rf'$\alpha={a}$')
    plt.xlabel(r'$\Delta t$'); plt.ylabel(r'$L^\infty$ error'); plt.grid(True,which='both',alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'example1_temporal.png',dpi=300); plt.close()
    pd.DataFrame(rows).to_csv(OUT/'example1_spatial.csv',index=False); pd.DataFrame(temporal).to_csv(OUT/'example1_temporal.csv',index=False)
    return rows,temporal


def run_e2e3():
    rows2=[]; rows3=[]; temp2=[]; temp3=[]
    for a in E2_ALPHAS:
        cx=genocchi_coeffs_for_polynomial(polynomial_monomial_coeffs_from_product_power(2),max(E2_N_SPATIAL))
        e=[]
        for N in E2_N_SPATIAL:
            cN=genocchi_coeffs_for_polynomial(polynomial_monomial_coeffs_from_product_power(2),N)
            s=run_2d_example(N,E2_DT_SPATIAL,T_E2E3,a,a,1,1,lambda u: np.zeros_like(u),lambda u: np.zeros_like(u),e2_source_factory(a,a),cN,cN,100.,lambda X,Y,t:100*np.exp(-t)*p2(X)*p2(Y),'E2')
            e.append(s['Linf']); rows2.append({'example':2,'alpha':a,'N':N,'dt':E2_DT_SPATIAL,'T':T_E2E3,'Linf':s['Linf'],'L2':s['L2'],'CPU_s':s['CPU_s']})
            if a == E2_ALPHAS[-1] and N == 10:
                save_representative_2d_plot(s, 'example2', rf'Example 2 solution ($\alpha={a}$, $N={N}$)', 'u')
        plt.semilogy(E2_N_SPATIAL,e,'o-',label=rf'$\alpha={a}$')
        for dt in E2_DT_TEMP:
            cN=genocchi_coeffs_for_polynomial(polynomial_monomial_coeffs_from_product_power(2),E2_N_TEMP)
            r=run_2d_example(E2_N_TEMP,dt,T_E2E3,a,a,1,1,lambda u:np.zeros_like(u),lambda u:np.zeros_like(u),e2_source_factory(a,a),cN,cN,100.,lambda X,Y,t:100*np.exp(-t)*p2(X)*p2(Y),'E2')
            temp2.append((a,dt,r['Linf'],r['CPU_s']))
    plt.xlabel('N'); plt.ylabel(r'$L^\infty$ error'); plt.grid(True,which='both',alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'example2_spatial.png',dpi=300); plt.close()
    plt.figure()
    for a in E2_ALPHAS:
        z=[v for aa,dt,v,cpu in temp2 if aa==a]; plt.loglog(E2_DT_TEMP,z,'o-',label=rf'$\alpha={a}$')
    plt.xlabel(r'$\Delta t$'); plt.ylabel(r'$L^\infty$ error'); plt.grid(True,which='both',alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'example2_temporal.png',dpi=300); plt.close()
    df2=pd.DataFrame(rows2); df2.to_csv(OUT/'example2_spatial.csv',index=False)
    tdf2=pd.DataFrame([{'alpha':a,'dt':dt,'Linf':e,'CPU_s':cpu} for a,dt,e,cpu in temp2]); tdf2['order']=tdf2.groupby('alpha')['Linf'].transform(lambda s: convergence_order(list(s),E2_DT_TEMP)); tdf2.to_csv(OUT/'example2_temporal.csv',index=False)

    for a in E3_ALPHAS:
        e=[]
        for N in E3_N_SPATIAL:
            cN=genocchi_coeffs_for_polynomial(polynomial_monomial_coeffs_from_product_power(2),N)
            r=run_2d_example(N,E2_DT_SPATIAL,T_E2E3,a,a,.5,.75,lambda u:-.1*u*(1-u),lambda u:-.1*(1-2*u),e3_source_factory(a,a),cN,cN,100.,lambda X,Y,t:100*np.exp(-t)*p2(X)*p2(Y),'E3')
            e.append(r['Linf']);
            if a == E3_ALPHAS[-1] and N == 10:
                save_representative_2d_plot(r, 'example3', rf'Example 3 solution ($\alpha={a}$, $N={N}$)', 'u')
            rows3.append({'example':3,'alpha':a,'N':N,'dt':E2_DT_SPATIAL,'T':T_E2E3,'Linf':r['Linf'],'L2':r['L2'],'CPU_s':r['CPU_s']})
        plt.semilogy(E3_N_SPATIAL,e,'o-',label=rf'$\alpha={a}$')
        for dt in E3_DT_TEMP:
            cN=genocchi_coeffs_for_polynomial(polynomial_monomial_coeffs_from_product_power(2),E3_N_TEMP)
            r=run_2d_example(E3_N_TEMP,dt,T_E2E3,a,a,.5,.75,lambda u:-.1*u*(1-u),lambda u:-.1*(1-2*u),e3_source_factory(a,a),cN,cN,100.,lambda X,Y,t:100*np.exp(-t)*p2(X)*p2(Y),'E3')
            temp3.append((a,dt,r['Linf'],r['CPU_s']))
    plt.xlabel('N'); plt.ylabel(r'$L^\infty$ error'); plt.grid(True,which='both',alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'example3_spatial.png',dpi=300); plt.close()
    plt.figure()
    for a in E3_ALPHAS:
        z=[v for aa,dt,v,cpu in temp3 if aa==a]; plt.loglog(E3_DT_TEMP,z,'o-',label=rf'$\alpha={a}$')
    plt.xlabel(r'$\Delta t$'); plt.ylabel(r'$L^\infty$ error'); plt.grid(True,which='both',alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'example3_temporal.png',dpi=300); plt.close()
    pd.DataFrame(rows3).to_csv(OUT/'example3_spatial.csv',index=False)
    tdf3=pd.DataFrame([{'alpha':a,'dt':dt,'Linf':e,'CPU_s':cpu} for a,dt,e,cpu in temp3]); tdf3['order']=tdf3.groupby('alpha')['Linf'].transform(lambda s: convergence_order(list(s),E3_DT_TEMP)); tdf3.to_csv(OUT/'example3_temporal.csv',index=False)
    return rows2,temp2,rows3,temp3

# ================================================================
# Example 4: Allen-Cahn
# ================================================================

def p5(x): return x**5*(1-x)**5

def rp5(x,a):
    # derivative of x^5(1-x)^5 via its monomial expansion
    out=np.zeros_like(np.asarray(x,float))
    coeff=polynomial_monomial_coeffs_from_product_power(5)
    for m,cm in enumerate(coeff):
        if cm!=0: out += cm*riesz_monomial(x,m,a)
    return out

def e4_source_factory(ax,ay,kappa=.5):
    def src(x,y,t):
        s=p5(x)*p5(y); u=s*(1+t*t)
        diff=kappa*(1+t*t)*(rp5(x,ax)*p5(y)+p5(x)*rp5(y,ay))
        return 2*t*s-diff-u+u**3
    return src

def run_e4():
    rows=[]; temps=[]
    for a in E4_ALPHAS:
        e=[]
        for N in E4_N_SPATIAL:
            r=run_2d_example(N,E4_DT_SPATIAL,T_E4,a,a,.5,.5,lambda u:-u+u**3,lambda u:-1+3*u*u,e4_source_factory(a,a,.5),None,None,1.,lambda X,Y,t:p5(X)*p5(Y)*(1+t*t),'E4')
            e.append(r['Linf']);
            if a == E4_ALPHAS[-1] and N == 11:
                save_representative_2d_plot(r, 'example4', rf'Example 4 solution ($\alpha={a}$, $N={N}$)', 'u')
            rows.append({'example':4,'alpha':a,'N':N,'dt':E4_DT_SPATIAL,'T':T_E4,'Linf':r['Linf'],'L2':r['L2'],'CPU_s':r['CPU_s']})
        plt.semilogy(E4_N_SPATIAL,e,'o-',label=rf'$\alpha={a}$')
        for dt in E4_DT_TEMP:
            r=run_2d_example(E4_N_TEMP,dt,T_E4,a,a,.5,.5,lambda u:-u+u**3,lambda u:-1+3*u*u,e4_source_factory(a,a,.5),None,None,1.,lambda X,Y,t:p5(X)*p5(Y)*(1+t*t),'E4')
            temps.append((a,dt,r['Linf'],r['CPU_s']))
    plt.xlabel('N'); plt.ylabel(r'$L^\infty$ error'); plt.grid(True,which='both',alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'example4_spatial.png',dpi=300); plt.close()
    plt.figure(figsize=(8,5))
    for a in E4_ALPHAS:
        z=[v for aa,dt,v,cpu in temps if aa==a]; plt.semilogy(E4_DT_TEMP,z,'o-',label=rf'$\alpha={a}$')
    plt.xlabel(r'$\Delta t$'); plt.ylabel(r'$L^\infty$ error'); plt.title('Allen--Cahn temporal exactness check'); plt.grid(True,which='both',alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'example4_time_exactness.png',dpi=300); plt.close()
    pd.DataFrame(rows).to_csv(OUT/'example4_spatial.csv',index=False)
    td=pd.DataFrame([{'alpha':a,'dt':dt,'Linf':e,'CPU_s':cpu} for a,dt,e,cpu in temps]); td['order']=td.groupby('alpha')['Linf'].transform(lambda s: convergence_order(list(s),E4_DT_TEMP)); td.to_csv(OUT/'example4_temporal.csv',index=False)
    return rows,temps

# ================================================================
# Conditioning study
# ================================================================

def run_conditioning():
    rows=[]
    for a in [1.2,1.5,1.8]:
        for N in [6,8,10,12,14,16]:
            cstart=time.perf_counter()
            s=setup_1d(N,a)
            cpu_setup=time.perf_counter()-cstart
            # representative stable Jacobian at t=0, dt=.005 using Example 1 exact state
            u=s['x']**2*(1-s['x'])**2
            y0=s['Q'].T@u
            u_int=s['Q'][s['idx'],:]@y0
            Qint=s['Q'][s['idx'],:]; Qb=s['Q'][[0,-1],:]
            A=np.vstack([Qint/.005-.5*s['B']+.5*e1_reaction_prime(u_int)[:,None]*Qint,Qb])
            Araw=np.vstack([s['Phi'][s['idx'],:]/.005-.5*s['Rint']+.5*e1_reaction_prime(u_int)[:,None]*s['Phi'][s['idx'],:],s['Phi'][[0,-1],:]])
            rows.append({'alpha':a,'N':N,'cond_Phi':np.linalg.cond(s['Phi']),'cond_Riesz':np.linalg.cond(s['Rint']),'cond_A':np.linalg.cond(Araw),'cond_Jy':np.linalg.cond(A),'cond_B':np.linalg.cond(s['B']),'CPU_s':cpu_setup})
    df=pd.DataFrame(rows); df.to_csv(OUT/'conditioning.csv',index=False)
    for a in [1.2,1.5,1.8]:
        sub=df[df['alpha']==a].sort_values('N')
        fig, ax = plt.subplots(figsize=(8,5))
        ax.semilogy(sub['N'],sub['cond_A'],'o-',linewidth=1.8,label=r'$\kappa_2(A_c)$')
        ax.semilogy(sub['N'],sub['cond_Jy'],'s--',linewidth=1.8,label=r'$\kappa_2(J_y)$')
        ax.set_xlabel('N'); ax.set_ylabel('Condition number')
        ax.set_title(rf'Conditioning study ($\alpha={a}$)')
        ax.grid(True,which='both',alpha=.3); ax.legend(); fig.tight_layout()
        fig.savefig(FIG/f'conditioning_alpha_{str(a).replace('.','p')}.png',dpi=300,bbox_inches='tight'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5))
    for a in [1.2,1.5,1.8]:
        sub=df[df['alpha']==a].sort_values('N')
        ax.semilogy(sub['N'],sub['cond_A'],'o-',linewidth=1.6,label=rf'$A_c$, $\alpha={a}$')
        ax.semilogy(sub['N'],sub['cond_Jy'],'s--',linewidth=1.6,label=rf'$J_y$, $\alpha={a}$')
    ax.set_xlabel('N'); ax.set_ylabel('Condition number')
    ax.set_title('Raw versus QR-stabilized conditioning')
    ax.grid(True,which='both',alpha=.3); ax.legend(ncol=2,fontsize=8); fig.tight_layout()
    fig.savefig(FIG/'conditioning_all.png',dpi=300,bbox_inches='tight'); plt.close(fig)
    return df

# ================================================================
# Main driver
# ================================================================

def main():
    print('Running Example 1...')
    run_e1_studies()
    print('Running Examples 2 and 3...')
    run_e2e3()
    print('Running Example 4...')
    run_e4()
    print('Running conditioning study...')
    df=run_conditioning()
    df.to_csv(OUT/'conditioning_study.csv',index=False)
    generate_latex_tables()
    print('LaTeX tables generated.')
    print('Finished. Output:',OUT.resolve())

if __name__=='__main__': main()
