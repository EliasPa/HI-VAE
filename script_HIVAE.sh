#source ~/venv/bin/activate

declare dataset="ICU"
declare batch_size=100

declare m_perc=20
declare mask=1

#declare data_file=${dataset}/data.csv
#declare types_file=${dataset}/data_types.csv
#declare miss_file=${dataset}/Missing${m_perc}_${mask}.csv
#declare true_miss_file=${dataset}/MissingTrue.csv

# training
declare data_file=${dataset}/training_data_preprocessed_cleaned.csv
declare types_file=${dataset}/data_types.csv
declare miss_file=${dataset}/Missingxx_y.csv

# testing

declare test_data_file=${dataset}/test_data_preprocessed_cleaned.csv
declare test_miss_file=${dataset}/test_Missingxx_y.csv

declare model="model_HIVAE_inputDropout"
declare z_dim=2
declare y_dim=5
declare s_dim=10


train_model(){
    python main_scripts.py --model_name $1 --batch_size ${batch_size} --epochs 10 \
    --data_file ${data_file} --types_file ${types_file} --miss_file ${miss_file} \
    --dim_latent_z $2 --dim_latent_y $3 --dim_latent_s $4 \
    --save_file ${save_file} \
    #--true_miss_file ${true_miss_file}
}

test_model(){
    python main_scripts.py --model_name $1 --batch_size 10000000 --epochs 1 \
    --data_file ${test_data_file} --types_file ${types_file} --miss_file ${test_miss_file} \
    --dim_latent_z $2 --dim_latent_y $3 --dim_latent_s $4 \
    --save_file ${save_file} --train 0 --restore 1 \
    #--true_miss_file ${true_miss_file}
}


declare save_file=${model}_${dataset}_Missing${m_perc}_${mask}_z${z_dim}_y${y_dim}_s${s_dim}_batch${batch_size}
#declare test_save_file=${model}_${dataset}_Missing${m_perc}_${mask}_z${z_dim}_y${y_dim}_s${s_dim}_batch${batch_size}_test

train_model ${model} ${z_dim} ${y_dim} ${s_dim}
test_model ${model} ${z_dim} ${y_dim} ${s_dim}