if [ $2 = "is_first_stage" ]
then
  python main.py Charades --feature_type C3D --snapshot_pref $1 --$2 --gpu 1
else
  python main.py Charades --feature_type C3D --snapshot_pref $1 --$2 --resume $3 --gpu 1
fi
