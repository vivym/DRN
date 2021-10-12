if [ $2 = "is_first_stage" ]
then
  python main.py VidSTG --feature_type SparseTokens --snapshot_pref $1 --$2 --gpu 0
else
  python main.py VidSTG --feature_type SparseTokens --snapshot_pref $1 --$2 --resume $3 --gpu 0
fi
