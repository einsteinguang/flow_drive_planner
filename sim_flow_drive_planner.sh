export CUDA_VISIBLE_DEVICES=0,1,2,3

###################################
# User Configuration Section
###################################
export NUPLAN_DEVKIT_ROOT="path/to/nuplan-devkit"                 # nuplan-devkit absolute path (e.g., "/home/user/nuplan-devkit")
export NUPLAN_DATA_ROOT="path/to/nuplan/data"                     # nuplan dataset absolute path (e.g. "/data")
export NUPLAN_MAPS_ROOT="path/to/nuplan/maps"                     # nuplan maps absolute path (e.g. "/data/nuplan-v1.1/maps")
export NUPLAN_EXP_ROOT="path/to/save/experiments"                 # path to save nuplan and interplan experiments (e.g. "/data/nuplan-v1.1/exp")
export INTERPLAN_DEVKIT_ROOT="path/to/interPlan"                  # interPlan-devkit absolute path (e.g., ".../interPlan")
NUPLAN_EXP_DIR="$NUPLAN_EXP_ROOT"
INTERPLAN_EXP_DIR="$NUPLAN_EXP_ROOT/interplan_exp"

# if you use the provided pretrained model, set the following variable accordingly
CKPT_PATH="path/to/flow_drive_model.pth"                          # Path to the pretrained model checkpoint
# after training your own model, set the following variables accordingly, and set CKPT_PATH="None"
EPOCHS=(0)                                                        # List of epochs to test
MLFLOW_EXP_NAME="exp_name"                                        # Name of the mlflow experiment
RUN_NAME="run_name"                                               # Name of the saved mlflow run

POST_MODES=(0)                                                    # 0 for FlowDrive (no post-processing),
                                                                  # 1 for FlowDrive* with moderated guidance and post-processing
# List of challenges to test ("closed_loop_reactive_agents" "closed_loop_nonreactive_agents")
CHALLENGES=("closed_loop_reactive_agents" "closed_loop_nonreactive_agents")
# List of splits to test ("test14-random" "test14-hard" "val14" "interplan")
SPLITS=("test14-random" "test14-hard" "val14" "interplan")
render=false                                                      # Set to true to enable rendering and video generation
video_dir="path/to/save/videos"                                   # Directory to save videos if rendering is enabled
####################################
# End of User Configuration Section
###################################

mkdir -p "$NUPLAN_EXP_ROOT"
# Loop through each split, challenge, and epoch
for POST_MODE in "${POST_MODES[@]}"; do
    for EPOCH in "${EPOCHS[@]}"; do
        for CHALLENGE in "${CHALLENGES[@]}"; do
            for SPLIT in "${SPLITS[@]}"; do

                if [ "$CKPT_PATH" == "None" ]; then
                    echo "Using model from mlflow experiment: $MLFLOW_EXP_NAME, run: $RUN_NAME, epoch: $EPOCH"
                    FOLDER_NAME=${RUN_NAME}_${EPOCH}
                else
                    echo "Using model from checkpoint path: $CKPT_PATH"
                    FOLDER_NAME=ckpt_inference
                fi

                if [ "$render" = true ]; then
                    VIDEO_DIR="${video_dir}/${FOLDER_NAME}_${SPLIT}_${CHALLENGE}_post${POST_MODE}"
                    mkdir -p "$VIDEO_DIR"
                    echo "Video directory created at: $VIDEO_DIR"
                else
                    VIDEO_DIR=""
                fi

                # jump is split is interplan and challenge is closed_loop_nonreactive_agents
                if [ "$SPLIT" == "interplan" ] && [ "$CHALLENGE" == "closed_loop_nonreactive_agents" ]; then
                    echo "Skipping split: $SPLIT with challenge: $CHALLENGE"
                    continue
                fi
                echo "Processing split: $SPLIT, challenge: $CHALLENGE, epoch: $EPOCH"

                if [ "$SPLIT" == "interplan" ]; then
                    export NUPLAN_EXP_ROOT="$INTERPLAN_EXP_DIR"
                    mkdir -p "$NUPLAN_EXP_ROOT"
                else
                    export NUPLAN_EXP_ROOT="$NUPLAN_EXP_DIR"
                    mkdir -p "$NUPLAN_EXP_ROOT"
                fi

                if [[ "$SPLIT" == *"val14"* ]]; then
                    SCENARIO_BUILDER="nuplan"
                else
                    SCENARIO_BUILDER="nuplan_challenge"
                fi

                PLANNER=flow_drive
                if [ "$POST_MODE" -eq 1 ]; then
                    FOLDER_NAME=${FOLDER_NAME}_multi
                fi

                if [ "$SPLIT" != "interplan" ]; then
                    python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
                        +simulation=$CHALLENGE \
                        planner=$PLANNER \
                        planner.flow_drive.ckpt_path=$CKPT_PATH \
                        planner.flow_drive.mlflow_exp_name=$MLFLOW_EXP_NAME \
                        planner.flow_drive.load_run_name=$RUN_NAME \
                        planner.flow_drive.load_epoch=$EPOCH \
                        planner.flow_drive.post_mode=$POST_MODE \
                        planner.flow_drive.render=$render \
                        planner.flow_drive.video_dir=$VIDEO_DIR \
                        scenario_builder=$SCENARIO_BUILDER \
                        scenario_filter=$SPLIT \
                        experiment_uid=$PLANNER/$SPLIT/$FOLDER_NAME \
                        verbose=false \
                        worker=ray_distributed \
                        worker.threads_per_node=80 \
                        distributed_mode='SINGLE_NODE' \
                        number_of_gpus_allocated_per_simulation=0.1\
                        enable_simulation_progress_bar=true \
                        hydra.searchpath="[pkg://flow_drive.config.scenario_filter, \
                        pkg://flow_drive.config, \
                        pkg://nuplan.planning.script.config.common, \
                        pkg://nuplan.planning.script.experiments]"
                else
                    python $INTERPLAN_DEVKIT_ROOT/interplan/planning/script/run_simulation.py \
                        +simulation=default_interplan_benchmark \
                        planner=$PLANNER \
                        planner.flow_drive.ckpt_path=$CKPT_PATH \
                        planner.flow_drive.mlflow_exp_name=$MLFLOW_EXP_NAME \
                        planner.flow_drive.load_run_name=$RUN_NAME \
                        planner.flow_drive.load_epoch=$EPOCH \
                        planner.flow_drive.post_mode=$POST_MODE \
                        planner.flow_drive.render=$render \
                        planner.flow_drive.video_dir=$VIDEO_DIR \
                        scenario_filter=interplan10 \
                        experiment_name=$FOLDER_NAME \
                        verbose=true \
                        worker=ray_distributed \
                        worker.threads_per_node=64 \
                        distributed_mode='SINGLE_NODE' \
                        number_of_gpus_allocated_per_simulation=0.1 \
                        enable_simulation_progress_bar=true \
                        hydra.searchpath="[pkg://flow_drive.config.scenario_filter, \
                        pkg://flow_drive.config, \
                        pkg://interplan.planning.script.config.common,\
                        pkg://interplan.planning.script.config.simulation,\
                        pkg://interplan.planning.script.experiments,\
                        pkg://nuplan.planning.script.config.common,\
                        pkg://nuplan.planning.script.config.simulation,\
                        pkg://nuplan.planning.script.experiments]"
                fi

                echo "Completed processing for split: $SPLIT, challenge: $CHALLENGE, epoch: $EPOCH"
            done
        done
    done
done

echo "All splits, challenges, and epochs processed successfully!"
