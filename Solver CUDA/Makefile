METOD := Metodo
RUNGE := RungeKutta
RUN_G := RungeKutta_graph
BASHF := AdamsBashford
BAS_G := AdamsBashford_graph
MOULT := AdamsMoulton

PROBP := Problema
PROB1 := simpleadvdiff1d
PROB2 := advdiff1d
PROB3 := brusselator1d
PROB4 := brusselator1d_shuffle
PROB5 := brusselator1d_grid
PROB6 := brusselator2d
PROB7 := brusselator2d_shuffle
PROB0 := prueba

DIR_H := Headers
DIR_O := Obj
DIR_S := Source
DIR_D := Datos

ALL_HEADERS := $(DIR_H)/$(METOD).h  $(DIR_H)/$(PROBP).h \
			   $(DIR_H)/$(RUNGE).h  $(DIR_H)/$(RUN_G).h  $(DIR_H)/$(BASHF).h $(DIR_H)/$(BAS_G).h \
			   $(DIR_H)/$(PROB3).h  $(DIR_H)/$(PROB4).h \
			   $(DIR_H)/$(PROB5).h  $(DIR_H)/$(PROB6).h  $(DIR_H)/$(PROB7).h

ALL_OBJ     := $(DIR_O)/$(METOD).o  $(DIR_O)/$(PROBP).o \
			   $(DIR_O)/$(RUNGE).o  $(DIR_O)/$(RUN_G).o  $(DIR_O)/$(BASHF).o $(DIR_O)/$(BAS_G).o \
			   $(DIR_O)/$(PROB3).o  $(DIR_O)/$(PROB4).o \
			   $(DIR_O)/$(PROB5).o  $(DIR_O)/$(PROB6).o  $(DIR_O)/$(PROB7).o
 
ALL_SRC     := $(DIR_S)/$(METOD).cu $(DIR_S)/$(PROBP).cu \
			   $(DIR_S)/$(RUNGE).cu $(DIR_S)/$(RUN_G).cu $(DIR_S)/$(BASHF).cu $(DIR_S)/$(BAS_G).cu \
			   $(DIR_S)/$(PROB3).cu $(DIR_S)/$(PROB4).cu \
			   $(DIR_S)/$(PROB5).cu $(DIR_S)/$(PROB6).cu $(DIR_S)/$(PROB7).cu
			   
ALL_DATA    := $(DIR_D)/datos0.txt   $(DIR_D)/datos1.txt 

CC := nvcc -rdc=true
LIBS := -L/usr/local/cuda/lib64 -lcuda -lcudart

NUM_PROBS := 1 2 3 4
NUM_HEBRAS := 128 256
TAMS := 100 200 500 1000 2000 5000 10000

# Creacion del ejecutable
create: solverCUDA

solverCUDA: $(DIR_O)/main.o $(ALL_OBJ)
	$(CC) -I$(DIR_H) $^ -O3 -o $@

$(DIR_O)/main.o: $(DIR_S)/main.cu
	$(CC) -I$(DIR_H) -c $< -O3 -o $@

$(ALL_OBJ): $(DIR_O)/%.o: $(DIR_S)/%.cu $(DIR_H)/%.h 
	$(CC) -I$(DIR_H) -c $< -O3 -o $@

# Ejecucion de problema con las variables introducidas
run:
	./solverCUDA $(metodo) $(orden) $(problema) $(hebras) $(tamvector) $(t0) $(tf) $(salto) 

#exportar con un método u otro
exporta:
	-rm $(DIR_D)/*.png
	cd ./Datos/; python3 visualiza1234.py datos0.txt
	cd ./Datos/; python3 visualiza1234.py datos1.txt


# create + run + exportacion de datos
completo: create run exporta

clean:
	-rm $(DIR_O)/*.o
	-rm ./solverCUDA

mrproper: clean
	-rm $(DIR_D)/*.txt
	-rm $(DIR_D)/*.png