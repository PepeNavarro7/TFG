METOD := Metodo
RUNGE := RungeKutta
RUN_G := RungeKutta_graph
BASHF := AdamsBashford
BAS_G := AdamsBashford_graph
MOULT := AdamsMoulton
MOU_G := AdamsMoulton_graph

PROBP := Problema
PROB1 := simpleadvdiff1d
PROB2 := simpleadvdiff1d_shuffle
PROB3 := advdiff1d
PROB4 := advdiff1d_shuffle
PROB5 := brusselator1d
PROB6 := brusselator1d_shuffle
PROB7 := brusselator1d_grid
PROB8 := brusselator2d
PROB9 := brusselator2d_shuffle
PROB0 := prueba

DIR_H := Headers
DIR_O := Obj
DIR_S := Source
DIR_D := Datos

ALL_HEADERS := $(DIR_H)/$(METOD).h  $(DIR_H)/$(PROBP).h  $(DIR_H)/$(RUNGE).h  $(DIR_H)/$(RUN_G).h \
			   $(DIR_H)/$(BASHF).h  $(DIR_H)/$(BAS_G).h  $(DIR_H)/$(MOULT).h  $(DIR_H)/$(MOU_G).h \
			   $(DIR_H)/$(PROB1).h  $(DIR_H)/$(PROB2).h  $(DIR_H)/$(PROB3).h  $(DIR_H)/$(PROB4).h \
			   $(DIR_H)/$(PROB5).h  $(DIR_H)/$(PROB6).h  $(DIR_H)/$(PROB7).h  $(DIR_H)/$(PROB8).h  $(DIR_H)/$(PROB9).h

ALL_OBJ     := $(DIR_O)/$(METOD).o  $(DIR_O)/$(PROBP).o  $(DIR_O)/$(RUNGE).o  $(DIR_O)/$(RUN_G).o \
			   $(DIR_O)/$(BASHF).o  $(DIR_O)/$(BAS_G).o  $(DIR_O)/$(MOULT).o  $(DIR_O)/$(MOU_G).o \
			   $(DIR_O)/$(PROB1).o  $(DIR_O)/$(PROB2).o  $(DIR_O)/$(PROB3).o  $(DIR_O)/$(PROB4).o \
			   $(DIR_O)/$(PROB5).o  $(DIR_O)/$(PROB6).o  $(DIR_O)/$(PROB7).o  $(DIR_O)/$(PROB8).o  $(DIR_O)/$(PROB9).o
 
ALL_SRC     := $(DIR_S)/$(METOD).cu $(DIR_S)/$(PROBP).cu $(DIR_S)/$(RUNGE).cu $(DIR_S)/$(RUN_G).cu \
			   $(DIR_S)/$(BASHF).cu $(DIR_S)/$(BAS_G).cu $(DIR_S)/$(MOULT).cu $(DIR_S)/$(MOU_G).cu \
			   $(DIR_S)/$(PROB1).cu $(DIR_S)/$(PROB2).cu $(DIR_S)/$(PROB3).cu $(DIR_S)/$(PROB4).cu \
			   $(DIR_S)/$(PROB5).cu $(DIR_S)/$(PROB6).cu $(DIR_S)/$(PROB7).cu $(DIR_S)/$(PROB8).cu $(DIR_S)/$(PROB9).cu
			   
ALL_DATA    := $(DIR_D)/datos0.txt   $(DIR_D)/datos1.txt 

CC := nvcc -rdc=true
LIBS := -L/usr/local/cuda/lib64 -lcuda -lcudart

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
	./solverCUDA $(metodo) $(orden) $(problema) $(hebras) $(tamvector) $(t0) $(tf) $(salto) $(modo)

runs:
	@./solverCUDA $(metodo) $(orden) $(problema) $(hebras) $(tamvector) $(t0) $(tf) $(salto) $(modo)

NUM_METODO := 1 2 3 4 5 6
NUM_ORDEN := 1 2 3 4 5
NUM_PROBLEMA := 1 2 3 4 5 6 7 8 9
num_prob1 := 1 3 5
num_prob2 := 8
NUM_HEBRAS := 256
tam_vec1 := 15000 20000 30000 40000 75000 125000 175000 750000 1000000
tam_vec2 := 25 75 125 150 175 225 250 300 400 600 700
SALTO := 5 6 7

batch1:
	@for p in $(num_prob1); do \
		for t in $(tam_vec1); do \
			echo "Ejecutando: metodo=4 orden=4/5 problema=$$p hebras=256 tamvector=$$t salto=6"; \
			$(MAKE) --no-print-directory runs metodo=4 orden=5 problema=$$p hebras=256 tamvector=$$t t0=0.0 tf=1.0 salto=6 modo=1; \
		done; \
    done

batch2:
	@for p in $(num_prob2); do \
		for t in $(tam_vec2); do \
			echo "Ejecutando: metodo=4 orden=4/5 problema=$$p hebras=256 tamvector=$$t salto=6"; \
			$(MAKE) --no-print-directory runs metodo=4 orden=5 problema=$$p hebras=256 tamvector=$$t t0=0.0 tf=1.0 salto=6 modo=1; \
		done; \
	done; \

background:
	@nohup $(MAKE) batch1 > background.log 2>&1 &

background_3:
	@nohup sh -c "$(MAKE) batch1 && $(MAKE) batch2" > background.log 2>&1 &

stop:
	@echo "Deteniendo procesos batch..."
	@pkill -f "make batch1" || true
	@pkill -f "make batch2" || true
	@pkill -f solverOMP || true
	@echo "Procesos detenidos."

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
	-rm ./background.log

mrproper: clean
	-rm $(DIR_D)/*.txt
	-rm $(DIR_D)/*.png