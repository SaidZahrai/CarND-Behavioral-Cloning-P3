echo 'Loss, Val_loss' > loss.csv
grep $1 stdout | awk '{ print $8 "," $11}' >> loss.csv
