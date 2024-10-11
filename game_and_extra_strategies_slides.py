import numpy as np

option = ['C', 'D']  # cooperate (C) or defect (D)


# define strategies:

def nice(opponent_choices):
    '''Does not defect first (can start defecting if the opponent starts first)'''
    if 'D' in opponent_choices:
        return np.random.choice(option)  # not sure if this should be 'D' or random
    else:
        return 'C'


def retaliatory(opponent_choices):
    '''Punishes defection by defecting in the next round'''
    if opponent_choices[-1] == 'D':
        return 'D'
    else:
        return 'C'


def forgiving(opponent_choices):
    # not sure about this one
    '''Returns to cooperation following cooperation of opponent (cooperates first, then chooses at random)'''
    if opponent_choices[-1] == 'C':
        return 'C'
    else:
        return np.random.choice(option)


def suspicious(opponent_choices):
    '''Does not cooperate until the other cooperates'''
    if 'C' in opponent_choices:
        return np.random.choice(option)  # not sure if this should be 'C' or random
    else:
        return 'D'


def generous(opponent_choices):
    '''Does not always retaliate at a (first) defection (chooses randomly if they will or not)'''
    if opponent_choices[-1] == 'D':
        return np.random.choice(option)
    else:
        return 'C'


def game(n, P1_strategy, P2_strategy):
    '''A basic version of the Prisoner's Dilemma, running for n games, given strategies for P1 and P2.'''

    option = ['C', 'D']

    # initial scores:
    P1 = 0
    P2 = 0

    # arrays of choices
    P1_choices = []
    P2_choices = []

    for i in range(n):
        # print(f'Round {i}:')
        if len(P1_choices) == 0:
            if P1_strategy == 'nice':
                choice1 = 'C'
            elif P1_strategy == 'suspicious':
                choice1 = 'D'
            elif P1_strategy == 'retaliatory' or P1_strategy == 'forgiving' or P1_strategy == 'generous':
                choice1 = np.random.choice(option)
        else:
            if P1_strategy == 'nice':
                choice1 = nice(P2_choices)
            elif P1_strategy == 'retaliatory':
                choice1 = retaliatory(P2_choices)
            elif P1_strategy == 'forgiving':
                choice1 = forgiving(P2_choices)
            elif P1_strategy == 'suspicious':
                choice1 = suspicious(P2_choices)
            elif P1_strategy == 'generous':
                choice1 = generous(P2_choices)
            else:
                choice1 = np.random.choice(option)

        P1_choices.append(choice1)
        # print(f'P1_choices: {P1_choices}')

        if len(P2_choices) == 0:
            if P2_strategy == 'nice':
                choice2 = 'C'
            elif P2_strategy == 'suspicious':
                choice2 = 'D'
            elif P2_strategy == 'retaliatory' or P2_strategy == 'forgiving' or P2_strategy == 'generous':
                choice2 = np.random.choice(option)
        else:
            if P2_strategy == 'nice':
                choice2 = nice(P1_choices)
            elif P2_strategy == 'retaliatory':
                choice2 = retaliatory(P1_choices)
            elif P2_strategy == 'forgiving':
                choice2 = forgiving(P1_choices)
            elif P2_strategy == 'suspicious':
                choice2 = suspicious(P1_choices)
            elif P2_strategy == 'generous':
                choice2 = generous(P1_choices)
            else:
                choice2 = np.random.choice(option)

        P2_choices.append(choice2)
        # print(f'P2_choices: {P2_choices}')

        if choice1 == 'C' and choice2 == 'C':
            P1 += 3
            P2 += 3
        elif choice1 == 'C' and choice2 == 'D':
            P1 += 0
            P2 += 5
        elif choice1 == 'D' and choice2 == 'C':
            P1 += 5
            P2 += 0
        elif choice1 == 'D' and choice2 == 'D':
            P1 += 1
            P2 += 1
    return P1, P2

def game_average_k_rounds(n, P1_strategy, P2_strategy, k):
    P1_total = 0
    P2_total = 0
    for i in range(1,k+1):
        P1, P2 = game(n, P1_strategy, P2_strategy)
        P1_total += P1
        P2_total += P2
        #print(P1_total, P2_total)
    P1_average = P1_total/k
    P2_average = P2_total / k
    return P1_average, P2_average

print(f"Nice & Nice: {game_average_k_rounds(10, 'nice', 'nice', 1000)}")
print(f"Nice & Retaliatory: {game_average_k_rounds(10, 'nice', 'retaliatory', 1000)}")
print(f"Nice & Forgiving: {game_average_k_rounds(10, 'nice', 'forgiving', 1000)}")
print(f"Nice & Suspicious: {game_average_k_rounds(10, 'nice', 'suspicious', 1000)}")
print(f"Nice & Generous: {game_average_k_rounds(10, 'nice', 'generous', 1000)}")

print(f"Retaliatory & Nice: {game_average_k_rounds(10, 'retaliatory', 'nice', 1000)}")
print(f"Retaliatory & Retaliatory: {game_average_k_rounds(10, 'retaliatory', 'retaliatory', 1000)}")
print(f"Retaliatory & Forgiving: {game_average_k_rounds(10, 'retaliatory', 'forgiving', 1000)}")
print(f"Retaliatory & Suspicious: {game_average_k_rounds(10, 'retaliatory', 'suspicious', 1000)}")
print(f"Retaliatory & Generous: {game_average_k_rounds(10, 'retaliatory', 'generous', 1000)}")

print(f"Forgiving & Nice: {game_average_k_rounds(10, 'forgiving', 'nice', 1000)}")
print(f"Forgiving & Retaliatory: {game_average_k_rounds(10, 'forgiving', 'retaliatory', 1000)}")
print(f"Forgiving & Forgiving: {game_average_k_rounds(10, 'forgiving', 'forgiving', 1000)}")
print(f"Forgiving & Suspicious: {game_average_k_rounds(10, 'forgiving', 'suspicious', 1000)}")
print(f"Forgiving & Generous: {game_average_k_rounds(10, 'forgiving', 'generous', 1000)}")

print(f"Suspicious & Nice: {game_average_k_rounds(10, 'suspicious', 'nice', 1000)}")
print(f"Suspicious & Retaliatory: {game_average_k_rounds(10, 'suspicious', 'retaliatory', 1000)}")
print(f"Suspicious & Forgiving: {game_average_k_rounds(10, 'suspicious', 'forgiving', 1000)}")
print(f"Suspicious & Suspicious: {game_average_k_rounds(10, 'suspicious', 'suspicious', 1000)}")
print(f"Suspicious & Generous: {game_average_k_rounds(10, 'suspicious', 'generous', 1000)}")

print(f"Generous & Nice: {game_average_k_rounds(10, 'generous', 'nice', 1000)}")
print(f"Generous & Retaliatory: {game_average_k_rounds(10, 'generous', 'retaliatory', 1000)}")
print(f"Generous & Forgiving: {game_average_k_rounds(10, 'generous', 'forgiving', 1000)}")
print(f"Generous & Suspicious: {game_average_k_rounds(10, 'generous', 'suspicious', 1000)}")
print(f"Generous & Generous: {game_average_k_rounds(10, 'generous', 'generous', 1000)}")



#Trying out all the options here:
#print(f"Nice & Nice: {game(10,'nice','nice')}")
#print(f"Nice & Retaliatory: {game(10,'nice','retaliatory')}")
#print(f"Nice & Forgiving: {game(10,'nice','forgiving')}")
#print(f"Nice & Suspicious: {game(10,'nice','suspicious')}")
#print(f"Nice & Generous: {game(10,'nice','generous')}")

#print(f"Retaliatory & Nice: {game(10,'retaliatory','nice')}")
#print(f"Retaliatory & Retaliatory: {game(10,'retaliatory','retaliatory')}")
#print(f"Retaliatory & Forgiving: {game(10,'retaliatory','forgiving')}")
#print(f"Retaliatory & Suspicious: {game(10,'retaliatory','suspicious')}")
#print(f"Retaliatory & Generous: {game(10,'retaliatory','generous')}")

#print(f"Forgiving & Nice: {game(10,'forgiving','nice')}")
#print(f"Forgiving & Retaliatory: {game(10,'forgiving','retaliatory')}")
#print(f"Forgiving & Forgiving: {game(10,'forgiving','forgiving')}")
#print(f"Forgiving & Suspicious: {game(10,'forgiving','suspicious')}")
#print(f"Forgiving & Generous: {game(10,'forgiving','generous')}")

#print(f"Suspicious & Nice: {game(10,'suspicious','nice')}")
#print(f"Suspicious & Retaliatory: {game(10,'suspicious','retaliatory')}")
#print(f"Suspicious & Forgiving: {game(10,'suspicious','forgiving')}")
#print(f"Suspicious & Suspicious: {game(10,'suspicious','suspicious')}")
#print(f"Suspicious & Generous: {game(10,'suspicious','generous')}")

#print(f"Generous & Nice: {game(10,'generous','nice')}")
#print(f"Generous & Retaliatory: {game(10,'generous','retaliatory')}")
#print(f"Generous & Forgiving: {game(10,'generous','forgiving')}")
#print(f"Generous & Suspicious: {game(10,'generous','suspicious')}")
#print(f"Generous & Generous: {game(10,'generous','generous')}")