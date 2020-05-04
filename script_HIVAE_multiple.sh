#source ./env/Scripts/activate

declare dataset="ICU"
declare m_perc=20
declare mask=1

# training
declare data_file=${dataset}/training_data_preprocessed_cleaned.csv
declare types_file=${dataset}/data_types.csv
declare miss_file=${dataset}/Missingxx_y.csv

# testing
declare test_data_file=${dataset}/test_data_preprocessed_cleaned.csv
declare test_miss_file=${dataset}/test_Missingxx_y.csv

declare model="model_HIVAE_inputDropout"

train_model(){
    python main_scripts.py --model_name $1 --batch_size $5 --epochs ${10} \
    --data_file $6 --types_file $7 --miss_file $8 \
    --dim_latent_z $2 --dim_latent_y $3 --dim_latent_s $4 \
    --save_file $9
}

test_model(){
    python main_scripts.py --model_name $1 --batch_size 10000000 --epochs 1 \
    --data_file $5 --types_file $6 --miss_file $7 \
    --dim_latent_z $2 --dim_latent_y $3 --dim_latent_s $4 \
    --save_file $8 --train 0 --restore 1
}

# train and test first model
declare z_dim=2
declare y_dim=5
declare s_dim=10
declare batch_size=128
declare epochs=200
declare save_file=${model}_${dataset}_Missing${m_perc}_${mask}_z${z_dim}_y${y_dim}_s${s_dim}_batch${batch_size}_epochs${epochs}

train_model ${model} ${z_dim} ${y_dim} ${s_dim} ${batch_size} ${data_file} ${types_file} ${miss_file} ${save_file} ${epochs}
test_model ${model} ${z_dim} ${y_dim} ${s_dim} ${test_data_file} ${types_file} ${test_miss_file} ${save_file}

# train and test second model
z_dim=2
y_dim=5
s_dim=10
epochs=400
batch_size=128
save_file=${model}_${dataset}_Missing${m_perc}_${mask}_z${z_dim}_y${y_dim}_s${s_dim}_batch${batch_size}_epochs${epochs}

train_model ${model} ${z_dim} ${y_dim} ${s_dim} ${batch_size} ${data_file} ${types_file} ${miss_file} ${save_file} ${epochs}
test_model ${model} ${z_dim} ${y_dim} ${s_dim} ${test_data_file} ${types_file} ${test_miss_file} ${save_file}

# train and test third model
z_dim=2
y_dim=5
s_dim=10
epochs=800
batch_size=128
save_file=${model}_${dataset}_Missing${m_perc}_${mask}_z${z_dim}_y${y_dim}_s${s_dim}_batch${batch_size}_epochs${epochs}

train_model ${model} ${z_dim} ${y_dim} ${s_dim} ${batch_size} ${data_file} ${types_file} ${miss_file} ${save_file} ${epochs}
test_model ${model} ${z_dim} ${y_dim} ${s_dim} ${test_data_file} ${types_file} ${test_miss_file} ${save_file}
