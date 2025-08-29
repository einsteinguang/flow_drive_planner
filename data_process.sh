###################################
# User Configuration Section
###################################
NUPLAN_DATA_PATH="/path/to/nuplan-v1.1/splits/trainval" # nuplan training data path (e.g., "/data/nuplan-v1.1/splits/trainval")
NUPLAN_MAP_PATH="/path/to/maps" # nuplan map path (e.g., "/data/nuplan-v1.1/maps")

TRAIN_SET_PATH="/path/to/save/processed/data/" # specify a folder to save the processed training data (will be around 150GB for 1M training data)
###################################

python data_process.py \
--data_path $NUPLAN_DATA_PATH \
--map_path $NUPLAN_MAP_PATH \
--save_path $TRAIN_SET_PATH \
--total_scenarios 1000000 \

