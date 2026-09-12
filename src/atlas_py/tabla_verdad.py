# | P     | Q     |
# | ----- | ----- |
# | True  | True  |
# | True  | False |
# | False | True  |
# | False | False | {COMBINACUONES POSIBLES BASE}

# ¬, ∧, v, →, ↔

# negacion_p = not P
# negacion_q = not Q
# conjuncion = P and Q
# disyuncion = P or Q
# condicional = (not P) or Q (Es la equivalencia lógica de -> entonces)
# bicondicional = P == Q (Solo es verdadero cuando ambos tienen el mismo valor)

    # P = True
    # Q = False

    # #Calcular
    # negativo_p = not P
    # negativo_q = not Q
    # p_y_q = P and Q 
    # p_o_q = P or Q
    # p_entonces_q = (not P) or Q
    # p_solo_si_q = P == Q

    # print("P: ", P)
    # print("Q: ", Q)
    # print("¬ P: ", negativo_p)
    # print("¬ Q: ", negativo_q)
    # print("P ∧ Q: ", p_y_q)
    # print("P v Q: ", p_o_q)
    # print("P → Q: ", p_entonces_q)
    # print("P ↔ Q: ", p_solo_si_q)

#Crear combinaciones, en lista y tuplas 

COMBINACIONES = [
    (True, True),
    (True, False),
    (False, True),
    (False, False)
    ]

print(f"| {'P':^7} | {'Q':^7} | {'¬P':^7} | {'¬Q':^7} | {'P∧Q':^7} | {'PvQ':^7} | {'P→Q':^7} | {'P↔Q':^7} |")

for P, Q in COMBINACIONES:
    negativo_p = not P
    negativo_q = not Q
    p_y_q = P and Q 
    p_o_q = P or Q
    p_entonces_q = (not P) or Q
    p_solo_si_q = P == Q

    print(f"| {str(P):^7} | {str(Q):^7} | {str(negativo_p):^7} | {str(negativo_q):^7} | {str(p_y_q):^7} | {str(p_o_q):^7} | {str(p_entonces_q):^7} | {str(p_solo_si_q):^7} |")#F-strings y str() para conservar el valor de booleanos

print(f"{'':/^100}")

#Crear combinaciones, en lista y tuplas 

COMBINACIONES = [
    (True, True),
    (True, False),
    (False, True),
    (False, False)
    ]

print(f"| {'P':^7} | {'Q':^7} | {'¬P':^7} | {'¬Q':^7} | {'¬P∧Q':^7} | {'Pv¬Q':^7} | {'Q→P':^7} |")

for P, Q in COMBINACIONES:
    negativo_p = not P
    negativo_q = not Q
    not_p_y_q = (not P) and Q 
    p_o_not_q = P or (not Q)
    q_entonces_p = (not Q) or P

    print(f"| {str(P):^7} | {str(Q):^7} | {str(negativo_p):^7} | {str(negativo_q):^7} | {str(not_p_y_q):^7} | {str(p_o_not_q):^7} | {str(q_entonces_p):^7} |")

print(f"{'':/^100}")

COMBINACIONES = [
    (True, True),
    (True, False),
    (False, True),
    (False, False)
    ]

print(f"| {'P':^7} | {'Q':^7} | {'¬P':^7} | {'P ∧ Q':^7} | {'¬(P∧Q)':^7} | {'¬Pv¬Q':^7} |")

for P, Q in COMBINACIONES:
    negativo_p = not P
    p_y_q = P and Q
    not_p_y_q = not (P and Q) 
    not_p_o_not_q = (not P) or (not Q)

    print(f"| {str(P):^7} | {str(Q):^7} | {str(negativo_p):^7} | {str(p_y_q):^7} | {str(not_p_y_q):^7} | {str(not_p_o_not_q):^7} |")
