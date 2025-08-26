import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

def Plot3D(X, Y, V, r=6, filename='lyapunov_function.png'):     
    # Plot Lyapunov functions  
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, V, rstride=5, cstride=5, alpha=0.5, cmap=cm.coolwarm)
    ax.contour(X, Y, V, 10, zdir='z', offset=0, cmap=cm.coolwarm)
    
    # Plot Valid region computed by dReal
    theta = np.linspace(0, 2*np.pi, 50)
    xc = r*np.cos(theta)
    yc = r*np.sin(theta)
    ax.plot(xc[:],yc[:],'r',linestyle='--', linewidth=2 ,label='Valid region')
    plt.legend(loc='upper right')

    ax.set_xlabel('$\Theta$')
    ax.set_ylabel('$\dot{\Theta}$')
    ax.set_zlabel('V')
    plt.title('Lyapunov Function')
    plt.savefig(filename)
    plt.close()

def Plotflow(Xd, Yd, t, f):
    # Plot phase plane 
    DX, DY = f([Xd, Yd],t)
    DX=DX/np.linalg.norm(DX, ord=2, axis=1, keepdims=True)
    DY=DY/np.linalg.norm(DY, ord=2, axis=1, keepdims=True)
    plt.streamplot(Xd,Yd,DX,DY, color=('gray'), linewidth=0.5,
                  density=0.5, arrowstyle='-|>', arrowsize=1.5)

def plot_2D_roa_lie_overlay(X, Y, V_nn_est, f=None, r=None, filename='2d_roa.png',
                            positive_examples=None, negative_examples=None, borderline_examples=None):
    '''
    Plot Region of attraction for systems with 2 state variables
    '''
    fig = plt.figure(figsize=(8,6))

    ax = plt.gca()
    legend_list = []
    label_list = []
    # Plot positive and negative lie derivatives samples
    inside_contour = V_nn_est.reshape(10000) < 0
    # Plot values less than 0 in green
    x1, x2 = np.meshgrid(X, Y)
    x = np.vstack([x1.flatten(), x2.flatten()]).transpose(1, 0)
    theta = x[:, 0][inside_contour]
    theta_dot = x[:, 1][inside_contour]
    positive_examples = positive_examples[inside_contour]
    negative_examples = negative_examples[inside_contour]
    borderline_examples = borderline_examples[inside_contour]

    ax.scatter(theta[positive_examples], theta_dot[positive_examples], color='green', label=r'$L_V < 0$', s=5)
    legend_list.append(plt.Rectangle((0,0),1,2,color='green',fill=False,linewidth = 2))
    label_list.append(r'$L_V < 0$')

    # Plot values greater or equal to 0 in red
    ax.scatter(theta[negative_examples], theta_dot[negative_examples], color='red', label=r'$L_V > \epsilon$', s=5)
    legend_list.append(plt.Rectangle((0,0),1,2,color='red',fill=False,linewidth = 2))
    label_list.append(r'$L_V > \epsilon$')

    # Plot values greater or equal to 0 in red
    ax.scatter(theta[borderline_examples], theta_dot[borderline_examples], color='yellow', label=r'$0 \leq L_V \leq \epsilon$', s=5)
    legend_list.append(plt.Rectangle((0,0),1,2,color='yellow',fill=False,linewidth = 2))
    label_list.append(r'$0 \leq L_V \leq \epsilon$')
    # Vaild Region
    C = plt.Circle((0, 0), r, color='grey', linewidth=1.5, fill=False, linestyle='--')
    ax.add_artist(C)

    # plot direction field
    xd = np.linspace(-r, r, 10) 
    yd = np.linspace(-r, r, 10)
    Xd, Yd = np.meshgrid(xd,yd)
    t = np.linspace(0, 2, 100)
    Plotflow(Xd, Yd, t, f) 

    # plot contour of estimated lyapunov
    ax.contour(X, Y, V_nn_est, levels=0, linewidths=2, colors='k')
    legend_list.append(plt.Rectangle((0,0),1,2,color='k',fill=False,linewidth = 2))
    label_list.append('NN, True Loss')

    legend_list.append(C)
    label_list.append('Valid Region')

    plt.title('Region of Attraction')
    plt.legend(legend_list, label_list, loc='upper right')
    plt.xlabel(r'Angle, $\theta$ (rad)')
    plt.ylabel(r'Angular velocity $\dot{\theta}$')
    plt.savefig(filename)
    plt.close()

def plot_2D_roa(X, Y, V_lqr, V_nn_est, f=None, r=None, filename='2d_roa.png',
                V_nn_est2=None, V_nn_est3=None):
    '''
    Plot Region of attraction for systems with 2 state variables
    '''
    fig = plt.figure(figsize=(8,6))

    ax = plt.gca()
    # Vaild Region
    C = plt.Circle((0, 0), r, color='grey', linewidth=1.5, fill=False, linestyle='--')
    ax.add_artist(C)

    # plot direction field
    xd = np.linspace(-r, r, 10) 
    yd = np.linspace(-r, r, 10)
    Xd, Yd = np.meshgrid(xd,yd)
    t = np.linspace(0, 2, 100)
    Plotflow(Xd, Yd, t, f) 

    legend_list = []
    label_list = []

    # plot contour of estimated lyapunov
    ax.contour(X, Y, V_nn_est, levels=0, linewidths=2, colors='k')
    legend_list.append(plt.Rectangle((0,0),1,2,color='k',fill=False,linewidth = 2))
    label_list.append('NN, True Loss')

    # plot contour of lyapunov wehere function equals 2.6 V(x) = 2.6
    ax.contour(X, Y, V_lqr, levels=0, linewidths=2, colors='m', linestyles='--')
    legend_list.append(plt.Rectangle((0,0),1,2,color='m',fill=False,linewidth = 2,linestyle='--'))
    label_list.append('LQR')

    if V_nn_est2 is not None:
        ax.contour(X, Y, V_nn_est2, levels=0, linewidths=2, colors='blue')
        legend_list.append(plt.Rectangle((0,0),1,2,color='blue',fill=False,linewidth = 2))
        label_list.append('NN, Appx Dynamics Loss')

    if V_nn_est3 is not None:
        ax.contour(X, Y, V_nn_est3, levels=0, linewidths=2, colors='green')
        legend_list.append(plt.Rectangle((0,0),1,2,color='green',fill=False,linewidth = 2))
        label_list.append('NN, Appx Lie Derivative Loss')

    legend_list.append(C)
    label_list.append('Valid Region')

    plt.title('Region of Attraction')
    plt.legend(legend_list, label_list, loc='upper right')
    plt.xlabel(r'Angle, $\theta$ (rad)')
    plt.ylabel(r'Angular velocity $\dot{\theta}$')
    plt.savefig(filename)
    plt.close()