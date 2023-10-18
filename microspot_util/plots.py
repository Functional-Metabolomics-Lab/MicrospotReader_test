import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib as mpl
from .utility import *
import math
import pyopenms as oms
from pyopenms.plotting import plot_chromatogram,plot_spectrum

def plot_grid(figure,axs,img,lines):
    axs.imshow(img)
    for item in lines:
        axs.axline((0,item.y_int), slope=item.slope,c="r")
    axs.set(ylim=[img.shape[0],0],xlim=[0,img.shape[1]])
    axs.axis("off")
    axs.set(title="Detected Spot-Grid")

def plot_result(figure,axs,img,df,g_prop):
    axs.imshow(img)
    axs.scatter(df.loc[df["note"]=="Initial Detection","x_coord"],df.loc[df["note"]=="Initial Detection","y_coord"],marker="2",c="k",label="Kept Spots")
    axs.scatter(df.loc[df["note"]=="Backfilled","x_coord"],df.loc[df["note"]=="Backfilled","y_coord"],marker="2",c="r",label="Backfilled Spots")
    axs.set(title="Detected Spots and Halos",
        ylabel="Row",
        xlabel="Column",
        yticks=df[df["column"]==g_prop["columns"]["bounds"][0]]["y_coord"],
        yticklabels=df[df["column"]==g_prop["columns"]["bounds"][0]]["row_name"],
        xticks=df[df["row"]==g_prop["rows"]["bounds"][0]]["x_coord"],
        xticklabels=df[df["row"]==g_prop["rows"]["bounds"][0]]["column"],
        )
    axs.spines[["right","left","top","bottom"]].set_visible(False)
    axs.tick_params(axis=u'both', which=u'both',length=0,labelsize=7)

    # Adding legend handles for Text
    handles,labels=axs.get_legend_handles_labels()
    patch=mpl.patches.Patch(facecolor="white",edgecolor="k",linewidth=0.4,label="Halo Radii")
    handles.append(patch)
    box = axs.get_position()
    axs.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])
    axs.legend(handles=handles,loc='upper center', bbox_to_anchor=(0.5, -0.1),
            fancybox=True,ncol=5)

    # Displaying all detected Halos with their respective radii.
    halo_df=df[df["halo"]>0]
    for idx in halo_df.index:
        axs.text(halo_df.loc[idx,"x_coord"]+12, halo_df.loc[idx,"y_coord"]-9, f'{halo_df.loc[idx,"halo"]:.0f}',c="white",size=7,path_effects=[pe.withStroke(linewidth=1, foreground="k")])

def plot_heatmap(figure,axs,df,g_prop,norm_data:bool=False):
    if norm_data==False:
        heatmap=df.pivot_table(index="row_name",columns="column",values="spot_intensity")
        title="Heatmap of Spot-Intensities"

    elif norm_data==True: 
        heatmap=df.pivot_table(index="row_name",columns="column",values="norm_intensity")
        title="Heatmap of normalized Spot-Intensities"

    htmp=axs.pcolormesh(heatmap.iloc[::-1],edgecolors="white",linewidth=4)
    axs.set(title=title,
            aspect="equal",
            ylabel="Row",
            xlabel="Column",
            yticks=np.array(range(1,len(heatmap)+1))-0.5,
            xticks=np.array(range(1,len(heatmap.columns)+1))-0.5,
            yticklabels=heatmap.index[::-1],
            xticklabels=heatmap.columns
            )
    axs.spines[["right","left","top","bottom"]].set_visible(False)
    axs.tick_params(axis=u'both', which=u'both',length=0)
    axs.scatter(df.loc[df["halo"]>0,"column"]-(math.floor((g_prop["columns"]["bounds"][0])/10)*10)-0.5,g_prop["rows"]["bounds"][-1]+0.5-df.loc[df["halo"]>0,"row"],marker="s", c="red",label="Detected Halos")
    box = axs.get_position()
    axs.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
            fancybox=True,ncol=5)
    figure.colorbar(htmp,ax=axs,label="Spot-Intensity",shrink=0.5)

def plot_chromatogram(figure,axs,df,norm_data:bool=False):
    if norm_data==False:
        axs.plot(df["RT"],df["spot_intensity"],c="k",linewidth=1)
        ylabel="Spot-Intensity [a.u.]"

    elif norm_data==True:
        axs.plot(df["RT"],df["norm_intensity"],c="k",linewidth=1)
        ylabel="Normalized Spot-Intensity [a.u.]"

    axs.set(
        title="Bioactivity-Chromatogram",
        ylabel=ylabel,
        xlabel="Time [s]",
        xlim=[df["RT"].min(),df["RT"].max()]
        )

def plot_mzml_chromatogram(figure,axs,exp,mz_val):
    rt_list=[]
    int_list=[]
    for spec in exp:
        if spec.getMSLevel() ==1:
            rt_list.append(spec.getRT())            
            peaks=np.array(spec.get_peaks())
            spot_int=peaks[1,np.where(peaks[0]==mz_val)][0]
            int_list.append(spot_int)

    axs.plot(rt_list,int_list,c="k",linewidth=1)
    
    axs.set(
        title="Chromatogram of Bioactivity",
        ylabel=f"Intensity @ m/z of {mz_val} [a.u.]",
        xlabel="Retention Time [s]",
        xlim=[min(rt_list),max(rt_list)],
        )
