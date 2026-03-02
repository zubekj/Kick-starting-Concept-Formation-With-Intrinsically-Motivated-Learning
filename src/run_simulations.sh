for i in 0 1 2 3 4 5 6 7 8 9; do
  #echo "python SMMain.py -s 100$i -w -t 50000 --group unsupervised_touch_refact_parasite_cont --name unsupervised_touch_refact_parasite_cont_s100$i --load_weights ../unsupervised_touch_refact_s100$i/storage-parasite/000999/weights.npy" | batch
  #echo "python SMMain.py -s 100$i -w -x -t 50000 --group unsupervised_touch_new_predictor --name unsupervised_touch_new_predictor_s100$i -o base_match_sigma=3 -o match_sigma=3" | batch
  echo "python SMMain.py -g -s 100$i -w --parasite -t 50000 --group parasite_mi1 --name parasite_mi1_s100$i -o epochs=1500" | batch
  #echo "python SMMain.py -g -s 100$i -w -t 50000 --group parasite_mi_cont1 --load_weights ../parasite_mi1_s100$i/storage-parasite/000999/weights.npy --name parasite_mi_cont1_s100$i -o epochs=1500" | batch
done;
