user="leyizh"
submission_record="results/submission_record.csv"

num_jobs="$(($(wc -l < $submission_record) - 1))"
echo "total number of jobs:" $num_jobs

num_pending="$(squeue -u $user -t PD -h | wc -l)"
num_running="$(squeue -u $user -t R -h | wc -l)"
echo "number of jobs pending:" $num_pending
echo "number of jobs running" $num_running
if [[ $((num_pending + num_running)) > 0 ]]; then 
	echo "some jobs still unfinished"
fi

num_success="$(squeue -u $user -t CD -h | wc -l)"
echo "num_success", $num_success
if [[ $num_success == $num_jobs ]]; then 
	echo "all jobs completed!"
fi

for ((i=2; i<=2; i++)); do
    parameter_id="$(awk -v i="$i" -F',' 'NR==i{print $1}' $submission_record)"
    realization="$(awk -v i="$i" -F',' 'NR==i{print $2}' $submission_record)"
    job="$(awk -v i="$i" -F',' 'NR==i{print $3}' $submission_record)"
    seed="$(awk -v i="$i" -F',' 'NR==i{print $4}' $submission_record)"
    
    # return job status
    job_status="$(squeue -j $job -o "%T" -h)"
    mem_old="$(squeue -j $job -o "%m" -h)"
    time_old="$(squeue -j $job -o "%l" -h)"
    echo $job_status, $mem_old, $time_old
    
    # compute new memory request
    mem_unit="$(echo $mem_old | grep -oE "[A-Za-z]")"
    mem_old_num="$(echo $mem_old | grep -oE "[0-9]")"
    mem_new_num=$((2 * mem_old_num))
    mem_new="$mem_new_num$mem_unit"
    echo $mem_new
    
    # compute new time request
    time_old_num="$(echo $time_old | cut -d: -f1)"
    time_new_num="$(printf "%02d" $((2 * time_old_num)))"
    time_new="$time_new_num:00"
    echo $time_new

    # job_submit="$(sbatch --time=$time_new job_submit_template.sh $parameter_id $realization $seed)"
    # job_submit_out="$(sbatch "${job_submit}")"
    # job_new="$(echo "${job_submit_out}" | grep -oE "[0-9]")"
done
