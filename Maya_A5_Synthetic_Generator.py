"""A5 Team 4: disclosed synthetic generation from Maya's 40 real tracking days.
Run: python Maya_A5_Synthetic_Generator.py
Input: Maya_A2_Model_Data(1).csv (placed alongside this script).
Outputs: Maya_A5_750_SYNTHETIC.csv and Maya_A5_Verification.txt.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
src = pd.read_csv(ROOT / 'Maya_A2_Model_Data(1).csv')
assert len(src)==40 and src['Date'].nunique()==40
N=750
rng=np.random.default_rng(42)
ycol='Daily_Steps'
continuous=['Steps_Before_Noon','Scheduled_School_Work_Hours','Rain_Before_Noon_mm']
binaries=['Weekend','Scheduled_Campus_Work_Day']
cols=[ycol]+continuous+binaries

def features(df, m0, h0):
    m=df.Steps_Before_Noon-m0
    h=df.Scheduled_School_Work_Hours-h0
    return sm.add_constant(pd.DataFrame({'M':m,'H':h,'MH':m*h,'Weekend':df.Weekend,
        'Campus':df.Scheduled_Campus_Work_Day,'Rain':df.Rain_Before_Noon_mm}),has_constant='add')

m0=src.Steps_Before_Noon.mean();h0=src.Scheduled_School_Work_Hours.mean()
Xorig=features(src,m0,h0)
model=sm.OLS(src[ycol],Xorig).fit()
# Joint-row resampling preserves the empirical associations, zero inflation and logical
# relationship between weekend and scheduled campus work. Numeric jitter prevents
# the 750 records from merely being exact copies of the original 40 days.
idx=rng.integers(0,len(src),size=N)
synth=src.iloc[idx].reset_index(drop=True).copy()
synth.insert(0,'Record_ID',[f'SYN-{k:04d}' for k in range(1,N+1)])
synth.insert(1,'Data_Origin','SYNTHETIC_SIMULATED_NOT_OBSERVED')
synth=synth.drop(columns=['Date'])
# Small jitter scaled to source SD; weather remains zero if source sampled a dry morning.
synth['Steps_Before_Noon']=np.maximum(0,np.rint(synth['Steps_Before_Noon'] + rng.normal(0,0.08*src.Steps_Before_Noon.std(ddof=1),N))).astype(int)
synth['Scheduled_School_Work_Hours']=np.maximum(0,np.round(synth['Scheduled_School_Work_Hours'] + rng.normal(0,0.12,N)* (synth['Scheduled_School_Work_Hours']>0),2))
wet=synth['Rain_Before_Noon_mm']>0
synth.loc[wet,'Rain_Before_Noon_mm']=np.maximum(0,np.round(synth.loc[wet,'Rain_Before_Noon_mm'] + rng.normal(0,0.15,wet.sum()),2))
# DGP: original fitted coefficients (including A3 interaction) plus sampled source
# regression residuals. No synthetic rows enter the fit of the real 40-row model.
expected=np.asarray(features(synth,m0,h0) @ model.params)
error_draw=rng.choice(np.asarray(model.resid), size=N, replace=True)
raw=np.rint(expected+error_draw)
# Walking counts must be nonnegative and cannot be below steps observed by noon.
# This physical constraint changes fitted synthetic moments and coefficients;
# the verification below reports the deviations rather than hiding them.
synth['Daily_Steps']=np.maximum(raw, synth['Steps_Before_Noon']).astype(int)
synth['Synthetic_Day_Index']=np.arange(1,N+1)
out=ROOT/'Maya_A5_750_SYNTHETIC.csv'
synth.to_csv(out,index=False)

refit=sm.OLS(synth[ycol],features(synth,m0,h0)).fit()
vars=[ycol,'Steps_Before_Noon','Scheduled_School_Work_Hours','Weekend','Scheduled_Campus_Work_Day','Rain_Before_Noon_mm']
lines=[]
lines.append('A5 TEAM 4 — MAYA PHYSICAL ACTIVITY, SYNTHETIC GENERATION AUDIT')
lines.append('Input: 40 REAL tracking-day rows. Output: 750 SYNTHETIC simulated rows. Seed: 42.')
lines.append('Synthetic data MUST NOT be represented as observed real activity.')
lines.append('Method: joint empirical resampling of predictors; continuous jitter; OLS A3 model with centered X1 and X2 and product; resampled observed residuals; physical step-count floor.')
lines.append('Uncertainty: 40-day source, correlated days, source-specific behaviors; preserves patterns approximately, not independent real evidence.')
lines.append('')
lines.append('ORIGINAL A3 COEFFICIENTS / SYNTHETIC REFIT')
for k in model.params.index:lines.append(f'{k:10s}: original {model.params[k]:12.6f}, synthetic {refit.params[k]:12.6f}')
lines.append('')
lines.append('MEAN AND SAMPLE SD (ddof=1)')
for c in vars:
 lines.append(f'{c}: original mean={src[c].mean():.4f}, SD={src[c].std(ddof=1):.4f}; synthetic mean={synth[c].mean():.4f}, SD={synth[c].std(ddof=1):.4f}')
lines.append('')
lines.append('PEARSON CORRELATIONS: ORIGINAL then SYNTHETIC')
lines.append(src[vars].corr().round(4).to_string())
lines.append('')
lines.append(synth[vars].corr().round(4).to_string())
lines.append('')
lines.append(f'Original: min daily={src[ycol].min()}, max daily={src[ycol].max()}; synthetic: min daily={synth[ycol].min()}, max daily={synth[ycol].max()}')
lines.append(f'Physical validity: all daily >= noon: {(synth[ycol]>=synth.Steps_Before_Noon).all()}; no negative counts: {(synth[ycol]>=0).all()}')
lines.append(f'All synthetic markers: {synth.Data_Origin.eq("SYNTHETIC_SIMULATED_NOT_OBSERVED").all()}')
lines.append('Note: Any reused source-like combinations and simulated noise are NOT new observed evidence; p-values on synthetic data are not confirmations of the original real-world effect.')
(ROOT/'Maya_A5_Verification.txt').write_text('\n'.join(lines))
print('\n'.join(lines))
print('Output:',out)
