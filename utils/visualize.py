import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from subprocess import Popen,PIPE


def draw_acf(acf): 
    plt.plot(np.arange(51),acf)
    plt.savefig('acf.png')

def draw_rate(wkdir,init):
    wkdir = f'result/rate/{wkdir}'
    for (p,r) in [(4,4)]:
    # for (p,r,dgp_s) in [(1,1,1),(0,1,1),(0,1,0),(0,0,1)]:
        df = pd.read_csv('{}/csv/p{}r{}.csv'.format(wkdir,p,r),header=None)
        df = df[df[1].str.contains(init)]
        T = [209,309,409,809]
        T0 = [9,9,9,9]
        s = [5,5,5,5]
        N=10
        # T1 = [100,200,300,500,700,1000]
        total_err = df[2]; est_err = df[3]; aprox_err = df[4]
        if not os.path.exists(f'{wkdir}/fig'):
            os.makedirs('{}/fig'.format(wkdir))
        
        plt.figure()
        plt.ylim([0,np.max(total_err)+0.1])
        x = np.divide(((r)*N+np.log(T0))*s,T)
        plt.xlim([0,np.max(x)+0.5])
        plt.plot(x,total_err,marker='o')
        plt.savefig(f'{wkdir}/fig/total_p{p}r{r}_{init}.png')

        plt.figure()
        plt.ylim([0,np.max(est_err)+0.1])
        x = np.divide(((r)*N+np.log(T0))*s,T)
        plt.xlim([0,np.max(x)+0.5])
        plt.plot(x,est_err,marker='o')
        plt.savefig(f'{wkdir}/fig/est_p{p}r{r}_{init}.png')

        plt.figure()
        plt.ylim([0,np.max(aprox_err)+0.1])
        x = np.divide(((r)*N+np.log(T0))*s,T)
        plt.xlim([0,np.max(x)+0.5])
        plt.plot(x,aprox_err,marker='o')
        plt.savefig(f'{wkdir}/fig/aprox_p{p}r{r}_{init}.png')

if __name__ == "__main__":     
    # draw_rate('GDversusTrue','true')
    draw_rate('season_AR','noisetrue')