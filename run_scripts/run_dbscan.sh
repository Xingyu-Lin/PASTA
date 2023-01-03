# replace dataset_path to your '<PATH TO DEMONSTRATION DATASET>' to generate dbscan labels for the dataset for pasta training
python core/pasta/generate_dbscan_label.py \
    --dataset_path '<PATH TO DEMONSTRATION DATASET>' \
    --dbscan_eps 0.03 \
    --dbscan_min_samples 6 \
    --dbscan_min_points 10