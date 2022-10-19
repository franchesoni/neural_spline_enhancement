TESTDIR="/home/franchesoni/adisk/datasets/mit5k/C/test/raw/"
TESTDIRTARGET="/home/franchesoni/adisk/datasets/mit5k/C/test/target/"
TESTLIST="/home/franchesoni/projects/current/neural_spline_enhancement/paper/processing/test_fnames.txt"
MODELPATH="/home/franchesoni/adisk/results/splines/models/expC.pth"
OUTDIRSPLINES="/home/franchesoni/adisk/results/splines/test_predictions/theirmodel/splines/"
# OUTDIR="/home/franchesoni/adisk/results/splines/test_predictions/theirmodel/"

OUTDIR="/home/franchesoni/adisk/results/splines/test_predictions/retraining/"

# python regen.py -i $TESTDIR -l $TESTLIST -md $MODELPATH -bs 1 -od $OUTDIR -ods $OUTDIRSPLINES


python test_images.py -i ${OUTDIR}/0/ -e $TESTDIRTARGET -l $TESTLIST -bs 1