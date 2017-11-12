# Make the binaries

all:
	nvcc -g -G test_alphabeta_c4.cu -o c4_out
	nvcc -g -G test_alphabeta_ttt.cu -o ttt_out

optimize:
	nvcc -03 test_alphabeta_c4.cu -o c4_out
	nvcc -03 test_alphabeta_ttt.cu -o ttt_out

clean:
	rm -f *_out a.out Outputs/*