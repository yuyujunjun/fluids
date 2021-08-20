# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import taichi as ti
import numpy as np
ti.init(arch=ti.cuda)
real=ti.f32
m=256
n=256
T = ti.field(dtype=real,shape=(m,n))# ambient
T_nxt = ti.field(dtype=real,shape=(m,n))# ambient
P = ti.field(dtype=real,shape=(m,n))# pressure
P_nxt = ti.field(dtype=real,shape=(m,n))# pressure
S = ti.field(dtype=real,shape=(m,n))# saturate
S_nxt = ti.field(dtype=real,shape=(m,n))# saturate
U = ti.Vector.field(2,dtype=real,shape=(m+1,n+1))
U_nxt = ti.Vector.field(2,dtype=real,shape=(m+1,n+1))
dx=1/n
dy=1/m
gui = ti.GUI("Smoke", (m, n), background_color=0x112F41)
dt=1/60


@ti.func
def boussinesq(s,T,alpha=1,beta=1):
    return ti.Vector([0,-alpha*s+beta*(T-0)*300])
@ti.func
def sample(qf, u, v):
    I = ti.Vector([int(u), int(v)])
    I = max(0, min(ti.Vector([m,n]) - 1, I))
    return qf[I]

@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)


@ti.func
def bilerp(vf, p,offset:float):
    u, v = p
    s, t = u - offset, v - offset
    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)
# 3rd order Runge-Kutta
@ti.func
def backtrace(vf, p,offset):
    v1 = bilerp(vf, p,offset)
    p1 = p - 0.5 * dt * v1
    v2 = bilerp(vf, p1,offset)
    p2 = p - 0.75 * dt * v2
    v3 = bilerp(vf, p2,offset)
    p -= dt * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
    return p
@ti.kernel
def advect(U:ti.template(),F:ti.template(),new_F:ti.template()):# for smoke we could assume that the velocity outside smoke region is zero,density is zero ,T is T_amb
    for i,j in F:
        pos=ti.Vector([i,j])+0.5
        pos = backtrace(U,pos,0.5)
        new_F[i,j]=bilerp(F,pos,0.5)
@ti.kernel
def advect_v():
    for i,j in U:
        pos=ti.Vector([i,j])
        pos = backtrace(U,pos,dt)
        U_nxt[i,j]=bilerp(U,pos,0.)
@ti.kernel
def add_force():
    for i,j in U:
        s=bilerp(S,ti.Vector([i,j]),0.5)
        t=bilerp(T,ti.Vector([i,j]),0.5)
        U[i,j]=U[i,j]+dt*boussinesq(s,t)


def integrate(U,U_nxt,T,T_nxt,S,S_nxt,P,P_nxt):
    add_force()
    advect(U,T,T_nxt)
    advect(U,S,S_nxt)
    advect(U,P,P_nxt)
    advect_v()

    U,U_nxt=U_nxt,U
    T,T_nxt=T_nxt,T
    S,S_nxt=S_nxt,S
    P,P_nxt=P_nxt,P

    #apply pressure
    

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    while True:
        while gui.get_event(ti.GUI.PRESS):
            pass
        dt=1/60
        if gui.is_pressed(ti.GUI.LMB):
            mxy = np.array(gui.get_cursor_pos(), dtype=np.float32) * np.array([n,m])
            y,x = mxy
            T[int(y),int(x)]=1
            S[int(y),int(x)]=0.5
            P[int(y),int(x)]=1


        integrate(U,U_nxt,T,T_nxt,S,S_nxt,P,P_nxt)
        gui.set_image(S)
        gui.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
