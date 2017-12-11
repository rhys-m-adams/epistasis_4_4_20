for ii in {0..1057}
do
sbatch --mem=40G scan_1.sh $ii
done

for ii in {0..1065}
do
sbatch --mem=40G scan_3.sh $ii 
done