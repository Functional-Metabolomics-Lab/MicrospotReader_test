import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib as mpl
from .utility import *
import math

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
    handles,lables=axs.get_legend_handles_labels()
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

def plot_heatmap(figure,axs,df,g_prop):
    heatmap=df.pivot_table(index="row_name",columns="column",values="spot_intensity")
    htmp=axs.pcolormesh(heatmap.iloc[::-1],edgecolors="white",linewidth=4)
    axs.set(title="Heatmap of Spot-Intensities",
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