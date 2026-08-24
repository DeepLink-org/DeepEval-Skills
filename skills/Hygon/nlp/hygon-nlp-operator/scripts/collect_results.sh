#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$SCRIPT_DIR/common.sh"
TARGET="${1:-all}"; require_choice TARGET "$TARGET" gemm gemm-conv longtail all
prepare_operator_dirs
python3 - "$TARGET" "$OPERATOR_RESULTS_DIR" "$OPERATOR_LOGS_DIR" <<'PY'
import csv,json,math,re,sys
from pathlib import Path
target,results_dir,logs_dir=sys.argv[1:]; results_dir=Path(results_dir); logs_dir=Path(logs_dir)
result={'status':'success','backend':'DTK-26.04/HIP-PyTorch-2.10','artifacts':{},'errors':[]}
names=[]
if target in {'gemm','gemm-conv','all'}: names += ['gemm_fp16','gemm_fp32']
if target in {'gemm-conv','all'}: names += ['conv_fp16','conv_fp32']
if target in {'longtail','all'}: names += ['longtail_fp16','longtail_fp32']
expected_rows={'gemm_fp16':224,'gemm_fp32':224,'conv_fp16':63,'conv_fp32':63,'longtail_fp16':40,'longtail_fp32':40}
for name in names:
 p=results_dir/f'{name}.csv'
 if not p.is_file(): result['errors'].append(name); continue
 with p.open(newline='',encoding='utf-8-sig') as stream: rows=list(csv.DictReader(stream))
 try: numeric_valid=all(math.isfinite(float(row['baseline'])) for row in rows)
 except (KeyError,TypeError,ValueError): numeric_valid=False
 valid=len(rows)==expected_rows[name] and numeric_valid; artifact={'path':str(p),'rows':len(rows),'valid':valid}
 if rows and numeric_valid: artifact['mean_baseline']=sum(float(row['baseline']) for row in rows)/len(rows)
 result['artifacts'][name]=artifact
 if not valid: result['errors'].append(name)
if target=='all':
 p=logs_dir/'transformer_block.log'; values=[float(v) for v in re.findall(r'Time per iteration of (?:encoder|decoder):\s*([0-9.eE+-]+)',p.read_text(errors='replace'))] if p.is_file() else []
 valid=len(values)==2 and all(math.isfinite(v) for v in values); result['artifacts']['transformer_block']={'path':str(p),'seconds_per_iteration':values,'valid':valid}
 if not valid: result['errors'].append('transformer_block')
if result['errors']: result['status']='failed'
(results_dir/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
if result['errors']: raise SystemExit(1)
PY
