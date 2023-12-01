import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import matplotlib as mpl
from .utility import *
import math
import pyopenms as oms
import string

def plot_grid(figure,axs,img,lines):
    axs.imshow(img)
    for item in lines:
        axs.axline((0,item.y_int), slope=item.slope,c="r")
    axs.set(ylim=[img.shape[0],0],xlim=[0,img.shape[1]])
    axs.axis("off")
    figure.tight_layout()

def plot_result(figure,axs,img,df,g_prop,halo:bool=True):
    axs.imshow(img)

    axs.scatter(
        df.loc[df["note"]=="Initial Detection","x_coord"],
        df.loc[df["note"]=="Initial Detection","y_coord"],
        marker="2",
        c="k",
        label="Kept Spots"
        )

    axs.scatter(
        df.loc[df["note"]=="Backfilled","x_coord"],
        df.loc[df["note"]=="Backfilled","y_coord"],
        marker="2",
        c="r",
        label="Backfilled Spots"
        )

    axs.set(
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

    # Adding halo specific items
    if halo is True:
        # Adding legend item for detected halos
        patch=mpl.patches.Patch(facecolor="white",edgecolor="k",linewidth=0.4,label="Halo Radii")
        handles.append(patch)

         # Displaying all detected Halos with their respective radii.
        halo_df=df[df["halo"]>0]
        for idx in halo_df.index:
            axs.text(halo_df.loc[idx,"x_coord"]+12, halo_df.loc[idx,"y_coord"]-9, f'{halo_df.loc[idx,"halo"]:.0f}',c="white",size=7,path_effects=[pe.withStroke(linewidth=1, foreground="k")])

    axs.legend(handles=handles,loc='upper center', bbox_to_anchor=(0.5, -0.1),
            fancybox=True,ncol=5)

    figure.tight_layout()

def plot_heatmapv2(figure,axs,df,conv_dict,norm_data=False,halo:bool=True):

    if norm_data is True:
        intensity="norm_intensity"
        colorbar_name="Normalized Spot Intensity [a.u.]"
    else:
        intensity="spot_intensity"
        colorbar_name="Spot Intensity [a.u.]"

    data=df.copy()
    data.loc[data["halo"]>0,"radius"]=data.loc[data["halo"]>0,"halo"]
    data["color"]="dimgray"
    data.loc[data.halo>0,"color"]="red"

    rad_factor=220/data.radius.max()

    sc=axs.scatter(data.column, -data.row, s=data.radius*rad_factor, c=data[intensity], cmap="viridis",edgecolors=data.color);
    axs.set(     
        aspect="equal",
        ylabel="Row",
        xlabel="Column",
        xticks=np.arange(data.column.min(),data.column.max()+1),
        yticks=-np.arange(data.row.min(),data.row.max()+1),
        yticklabels=[conv_dict[i].upper() for i in range(data.row.min(),data.row.max()+1)],
        xticklabels=np.arange(data.column.min(),data.column.max()+1)
        );

    axs.spines[["right","left","top","bottom"]].set_visible(False)
    axs.tick_params(axis=u'both', which=u'both',length=0)

    figure.colorbar(sc,shrink=0.7,label=colorbar_name,orientation="horizontal",location="top")

    if halo:
        marker_nohalo=Line2D([],[],color="white",markeredgecolor="dimgray",markerfacecolor="gray",marker="o")
        marker_halo=Line2D([],[],color="white",markeredgecolor="red",markerfacecolor="gray",marker="o")
        legend1 = axs.legend((marker_nohalo,marker_halo),("-","+"),title="Halo\nDetected",loc="upper left",bbox_to_anchor=(1,1),frameon=False)
        axs.add_artist(legend1)

        legendtitle="Radius of\nSpot or Halo"
    else:
        legendtitle="Radius of\nSpot"

    handles, labels = sc.legend_elements(prop="sizes", alpha=1,num=4,func=lambda x: x/rad_factor,color="dimgray",markeredgecolor="k")
    legend2 = axs.legend(handles, labels, loc="lower left", title=legendtitle,bbox_to_anchor=(1, 0),
                frameon=False)

    figure.tight_layout()

def plot_heatmap(figure,axs,df,g_prop,norm_data:bool=False):
    if norm_data==False:
        heatmap=df.pivot_table(index="row_name",columns="column",values="spot_intensity")

    elif norm_data==True: 
        heatmap=df.pivot_table(index="row_name",columns="column",values="norm_intensity")

    htmp=axs.pcolormesh(heatmap.iloc[::-1],edgecolors="white",linewidth=4)
    axs.set(
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
    # axs.set_position([box.x0, box.y0 + box.height * 0.1,
    #                 box.width, box.height * 0.9])
    axs.legend(loc='upper left', bbox_to_anchor=(1, 0),
            fancybox=True,ncol=5)
    figure.colorbar(htmp,ax=axs,label="Spot-Intensity",shrink=0.5)
    
    figure.tight_layout()

def plot_chromatogram(figure,axs,df,norm_data:bool=False):
    df.sort_values("RT",inplace=True)
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
    
    figure.tight_layout()

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
    figure.tight_layout()

def plot_activity_chromatogram(figure,axs,spot_df,peak_df,baseline_acceptance:float=0.02,ydata_name="norm_intensity"):
    std_old,mn_old=baseline_noise(spot_df[ydata_name],baseline_acceptance)

    axs.plot(spot_df.RT,spot_df[ydata_name],c="k",linewidth=1)
    axs.set(ylabel=f"{string.capwords(ydata_name.replace('_',' '))} [a.u.]",xlabel="Retention Time [s]")
    axs.scatter(peak_df.RT,peak_df.max_int,marker="x",c="red")
    
    axs.hlines([mn_old+3*std_old,mn_old-3*std_old],xmin=spot_df.RT.min(),xmax=spot_df.RT.max(),linewidth=1,colors="gray",ls="--")
    
    for idx in peak_df.index:
        axs.fill_between(spot_df.RT.loc[peak_df.loc[idx,"start_idx"]:peak_df.loc[idx,"end_idx"]],spot_df.loc[peak_df.loc[idx,"start_idx"]:peak_df.loc[idx,"end_idx"],ydata_name],color="lightblue")
        axs.text(peak_df.loc[idx,"RT"]*1.01, peak_df.loc[idx,"max_int"]*1.01, f'peak{idx}',c="k",size=7)
    axs.legend(["Chromatogram","Detected Peaks","Baseline-Noise"])

    figure.tight_layout()