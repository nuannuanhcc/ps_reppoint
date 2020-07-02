
# retinanet_r50_fpn_1x
# reppoints_moment_r50_fpn_1x
# reppoints_moment_r50_fpn_2x

export DIR="reppoints_moment_r50_fpn_1x"
export NAME="6_15_3"

for i in $(seq 1 12)
do
  let ITER=i
  python ./mmdetection/tools/test.py \
  ./configs/$DIR.py \
  "work_dirs/"$DIR"_"$NAME"/epoch_"$ITER".pth" \
  --out "work_dirs/"$DIR"_"$NAME"/results.pkl" --eval bbox
done