# prepare dataset

python generate_mc_test.py "${1}" "${2}"

python infer_mc.py
python infer_qa.py --output_file "${3}"
