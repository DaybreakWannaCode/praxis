"""Render the audited baseline without recomputing or selecting an endpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(source, output):
    raw=source.read_bytes(); result=json.loads(raw)
    if result['status']!='audited':raise ValueError('Audited paired analysis required')
    p=result['primary']; n=p['n']; t=p['transitions']
    assert sum(t.values())==n
    assert (t['1_to_0']+t['1_to_1'])/n==p['initial_accuracy']
    assert (t['0_to_1']+t['1_to_1'])/n==p['final_accuracy']
    assert (t['0_to_1']-t['1_to_0'])/n==p['accuracy_change']
    output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(8,3.5),gridspec_kw={'width_ratios':[1,1.4]})
    ax=axes[0]; values=[100*p['initial_accuracy'],100*p['final_accuracy']]
    bars=ax.bar(['Pretrained','Text RL'],values,color=['#7b8794','#315b80'],width=.55)
    ax.set_ylim(0,100);ax.set_ylabel('Greedy visual accuracy (%)')
    ax.set_title('A  Observed accuracy',loc='left',fontweight='bold')
    for bar,value in zip(bars,values):
        ax.text(bar.get_x()+bar.get_width()/2,value+2,f'{value:.2f}%',ha='center',fontsize=10)
    ax=axes[1]; delta=100*p['accuracy_change']; lo,hi=[100*v for v in p['paired_image_bootstrap_95']]
    ax.axvline(0,color='#a7a7a7',linestyle='--',linewidth=1)
    ax.errorbar(delta,0,xerr=[[delta-lo],[hi-delta]],fmt='o',color='#315b80',capsize=6,linewidth=2,markersize=7)
    limit=max(abs(lo),abs(hi),1)*1.35;ax.set_xlim(-limit,limit);ax.set_ylim(-1,1)
    ax.set_yticks([]);ax.spines['left'].set_visible(False)
    ax.set_xlabel('Paired accuracy change (percentage points)')
    ax.set_title('B  Gain remains unresolved',loc='left',fontweight='bold')
    ax.text(delta,.35,f'{delta:+.2f} pp  [95% interval: {lo:+.2f}, {hi:+.2f}]',ha='center',fontsize=9)
    ax.text(.5,.16,f"{t['0_to_1']} improved · {t['1_to_0']} worsened · {t['0_to_0']+t['1_to_1']} unchanged",
            transform=ax.transAxes,ha='center',fontsize=9)
    fig.subplots_adjust(left=.09,right=.98,bottom=.29,top=.86,wspace=.43)
    fig.text(.5,.065,f'{n} development images · one training seed · paired image-bootstrap interval\n'
             'Interval is conditional on this panel and trajectory; it does not capture between-seed variation.',
             ha='center',fontsize=8,color='#555555')
    for ext in ('png','pdf','svg'):fig.savefig(output/f'baseline.{ext}',dpi=220,facecolor='white')
    plt.close(fig)
    metadata={'analysis_sha256':hashlib.sha256(raw).hexdigest(),'plan_sha256':result['plan_sha256'],
              'primary':p,'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'matplotlib':matplotlib.__version__,
              'artifacts':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in output.iterdir()}}
    (output/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();render(args.source,args.output)
