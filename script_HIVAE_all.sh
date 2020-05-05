#source ~/venv/bin/activate
declare dataset="ICU"
declare n_variables=40
declare n_sample=8000
declare split=50


generate_data_types(){
    python data_types_generator.py ${n_variables}
}

generate_training_and_test_data(){
    python data_preprocessing.py ${n_variables} ${n_sample} ${split}
}

# Generate data_types-file
generate_data_types
# Generate training and test-data
generate_training_and_test_data


declare model="model_HIVAE_inputDropout"
declare batch_size=100
declare m_perc=20
declare mask=1
declare z_dim=2
declare y_dim=5
declare s_dim=10

declare data_path=${dataset}/dataset_${n_variables}_vars
# training
declare data_file=${data_path}/${n_sample}_N_${split}_split/training_data_preprocessed_cleaned.csv
declare types_file=${data_path}/data_types.csv
declare miss_file=${data_path}/${n_sample}_N_${split}_split//Missingxx_y.csv

# testing
declare test_data_file=${data_path}/${n_sample}_N_${split}_split/test_data_preprocessed_cleaned.csv
declare test_miss_file=${data_path}/${n_sample}_N_${split}_split/test_Missingxx_y.csv


train_model(){
    python main_scripts.py --model_name $1 --batch_size ${batch_size} --epochs ${5} \
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

evaluate_model(){
    python results_validation.py $1 $2 $3
}


declare save_file=${model}_${dataset}_${n_variables}_vars_z${z_dim}_y${y_dim}_s${s_dim}_batch${batch_size}
#declare test_save_file=${model}_${dataset}_Missing${m_perc}_${mask}_z${z_dim}_y${y_dim}_s${s_dim}_batch${batch_size}_test

splits=(50 70)
z_dims=(2 4)
y_dims=(5 7)
s_dims=(10 15)
epochs=(40 50)

echo ${z_dims[@]}
echo ${y_dims[@]}
echo ${s_dims[@]}
echo ${epochs[@]}

for z in ${z_dims[@]}
do
    for y in ${y_dims[@]}
    do 
        for s in ${s_dims[@]}
        do 
            for e in ${epochs[@]}
            do 
                echo ${z} ${y} ${s} ${e}
                save_file=${model}_${dataset}_${n_variables}_vars_z${z}_y${y}_s${s}_batch${batch_size}_epochs${e}
                train_model ${model} ${z} ${y} ${s} ${e}
                test_model ${model} ${z} ${y} ${s}
                evaluate_model ${save_file} ${data_path}/${n_sample}_N_${split}_split/test_target.csv ${n_variables}
                echo ${save_file}
            done
        done
    done
done
#train_model ${model} ${z_dim} ${y_dim} ${s_dim}
#test_model ${model} ${z_dim} ${y_dim} ${s_dim}