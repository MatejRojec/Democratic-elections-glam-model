import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def probability_n(r, previous_probability, k):
    if r % 2 == 1:  # Odd number
        m_start = (r + 1) // 2
        probability = sum(math.comb(r, m) * previous_probability**m * (1 - previous_probability)**(r - m) for m in range(m_start, r + 1))
    else:  # Even number
        m_start = r // 2 + 1
        tie_probability = math.comb(r, r // 2) * k * previous_probability**(r // 2) * (1 - previous_probability)**(r // 2)
        probability = sum(math.comb(r, m) * previous_probability**m * (1 - previous_probability)**(r - m) for m in range(m_start, r + 1)) + tie_probability

    return probability


def compounded_model(a1, a2, a3, a4, a5, a6, previous_probability, k):
    compounded_model = a1 * probability_n(1, previous_probability, k) + a2 * probability_n(2, previous_probability, k) + a3 * probability_n(3, previous_probability, k) + a4 * probability_n(4, previous_probability, k) + a5 * probability_n(5, previous_probability, k) + a6 * probability_n(6, previous_probability, k)
    return compounded_model


def find_values_near_fixed_points(compounded_model, k, a1= 1/6, a2= 1/6, a3= 1/6, a4= 1/6, a5= 1/6, a6= 1/6,step_size=0.00001, tolerance=1e-4):
    fixed_points = []
    for x in range(int(1/step_size)):
        current_value = x * step_size
        next_value = compounded_model(a1, a2, a3, a4, a5, a6, current_value, k)

        if abs(next_value - current_value) < tolerance and current_value > 0.001:
            fixed_points.append(current_value)
    return fixed_points[0]


# Example usage
values_near_fixed_points = find_values_near_fixed_points(compounded_model, 1)
print("Values near fixed points:", values_near_fixed_points)


r_values = [2, 3, 4, 5, 6]
p_values = [0.1, 0.3, 0.5, 0.7, 0.9]
k_values = [0, 0.5, 1]

for k in k_values:
    data = {'p_n': p_values}
    for r in r_values:
        probabilities = [probability_n(r, p, k) for p in p_values]
        data[f'r={r}'] = probabilities
    
    df = pd.DataFrame(data)
    print(f"\nTable for k={k}:\n")
    print(df)


k_value = 1 
r_value = 4

previous_probabilities = np.linspace(0, 1, 100)
current_probabilities = [probability_n(r_value, prev_prob, k_value) for prev_prob in previous_probabilities]

# Plotting
plt.plot(previous_probabilities, current_probabilities, label=f'k={k_value}, r={r_value}')
plt.plot(previous_probabilities, previous_probabilities, linestyle='--', label='y = x', color='red')
plt.xlabel('Previous Probability')
plt.ylabel('Current Probability')
plt.title(f'Probability Transition (k={k_value}, r={r_value})')
plt.legend()
plt.grid(True)
plt.show()


def derivative_probability_n(r, previous_probability, k):
    if r % 2 == 1:  # Odd number
        m_start = (r + 1) // 2
        probability = sum(math.comb(r, m) * previous_probability**m * (1 - previous_probability)**(r - m) for m in range(m_start, r + 1))
    else:  # Even number
        m_start = r // 2 + 1
        tie_probability = math.comb(r, r // 2) * k * previous_probability**(r // 2) * (1 - previous_probability)**(r // 2)
        probability = sum(math.comb(r, m) * previous_probability**m * (1 - previous_probability)**(r - m) for m in range(m_start, r + 1)) + tie_probability

    return round(probability, 4)


def calculate_fixed_point(k):
    numerator = 1 - 6 * k + np.sqrt(13 - 36 * k + 36 * k ** 2)
    denominator = 6 * (1 - 2 * k)
    result = numerator / denominator
    return result


# a_4 = 1 

k_values = np.linspace(0, 1, 1000)

fixed_points = [calculate_fixed_point(k) for k in k_values]

plt.figure(figsize=(8, 5))

plt.plot(k_values, fixed_points, label='Fixed Point')
plt.axhline(y=1/2, color='r', linestyle='--', label='y = 1/2')

count = 0
for k, fixed_point in zip(k_values, fixed_points):
    if count % 50 == 0 or count == 999:
        plt.arrow(k, fixed_point, 0, -fixed_point, color='blue', head_width=0.02, head_length=0.02)
        plt.arrow(k, fixed_point, 0, 1 - fixed_point, color='green', head_width=0.02, head_length=0.02)
    count = count + 1


plt.xlabel('k')
plt.ylabel('Tipping point')
plt.title('Tipping point calculation for $k$ in (0, 1), in the Galam model with $a_4$ = 1')

plt.legend(loc='upper left')

plt.show()

# 6 uniform groups

fixed_points_6 = [find_values_near_fixed_points(compounded_model, k) for k in k_values]

plt.figure(figsize=(8, 5))

plt.plot(k_values, fixed_points_6, label='Fixed Point')
plt.axhline(y=1/2, color='r', linestyle='--', label='y = 1/2')

count = 0
for k, fixed_point in zip(k_values, fixed_points_6):
    if count % 50 == 0 or count == 999:
        plt.arrow(k, fixed_point, 0, -fixed_point, color='blue', head_width=0.02, head_length=0.02)
        plt.arrow(k, fixed_point, 0, 1 - fixed_point, color='green', head_width=0.02, head_length=0.02)
    count = count + 1


plt.xlabel('k')
plt.ylabel('Tipping point')
plt.title('Tipping point calculation for $k$ in (0, 1), in the Galam model with 6 uniform groups')

plt.legend(loc='upper left')

plt.show()


# 5 uniform gropups uniform groups

fixed_points_5 = [find_values_near_fixed_points(compounded_model, k, 1/5, 1/5, 1/5, 1/5, 1/5, 0) for k in k_values]

plt.figure(figsize=(8, 5))

plt.plot(k_values, fixed_points_5, label='Fixed Point')
plt.axhline(y=1/2, color='r', linestyle='--', label='y = 1/2')

count = 0
for k, fixed_point in zip(k_values, fixed_points_5):
    if count % 50 == 0 or count == 999:
        plt.arrow(k, fixed_point, 0, -fixed_point, color='blue', head_width=0.02, head_length=0.02)
        plt.arrow(k, fixed_point, 0, 1 - fixed_point, color='green', head_width=0.02, head_length=0.02)
    count = count + 1


plt.xlabel('k')
plt.ylabel('Tipping point')
plt.title('Tipping point calculation for $k$ in (0, 1), in the Galam model with 5 uniform groups')

plt.legend(loc='upper left')

plt.show()

current_probability = 0.7
prob = [current_probability]
k = []
for _ in range(8):
    current_probability = compounded_model(1/5, 1/5, 1/5, 1/5, 1/5, 0, current_probability, 0)
    prob.append(current_probability)
    k.append(0)

prob.append(0.3)
k.append(1)

for _ in range(8):
    current_probability = compounded_model(1/5, 1/5, 1/5, 1/5, 1/5, 0, current_probability, 1)
    prob.append(current_probability)
    k.append(1)

for i in range(1, len(k)):
    color = 'red' if k[i - 1] == 0 else 'green'
    plt.plot([i-1, i], prob[i-1:i+1], color=color, label=f'k={k[i]}')

    if k[i] != k[i - 1]:
        plt.axvline(x=i, color='black', linestyle='--', linewidth=1)
        plt.text(i, 0.8, r'$k$ changed  from $0$ to $1$', ha='center')

legend_labels = [plt.Line2D([0], [0], color='green', lw=2, label='k=1'),
                 plt.Line2D([0], [0], color='red', lw=2, label='k=0')]
plt.legend(handles=legend_labels)

plt.xlabel('Steps')
plt.ylabel('Probability')
plt.title('Probability over time with change from $k=0$ to $k=1$')
plt.show()

# Probability over time with $k=0$ and $k=1$ in the Galam model with 5 uniform groups with starting probability p_0 = 

result = [fp6 > fp5 for fp6, fp5 in zip(fixed_points_6, fixed_points_5)]

print(result)

# Generalised model


def K(k, r, mu):
    return 1 + (k - 1) * int(r == 2 * mu)


def pi_0(p, k, r):
    result = 0
    for mu in range(int(r/2) + 1):
        result += math.comb(r, mu) * K(k, r, mu) * (p**(r - mu)) * ((1 - p)**mu)
    return result


def pi_1(p, k, r):
    result = 0
    for mu in range(int(r/2) + 1):
        result += math.comb(r, mu) * K(1 - k, r, mu) * (mu / (r * p)) * (p**(mu)) * ((1 - p)**(r - mu))
    return result


def pi_2(p, k, r):
    result = 0
    for mu in range(int(r/2) + 1):
        result += math.comb(r, mu) * K(k, r, mu) * (mu / (r * (1 - p))) * (p**(r - mu)) * ((1 - p)**mu)
    return result


def generalized_update(r, a, b, c, k, p):
    if p == 0:
        return 0
    if p == 1:
        return 1
    else:
        term_1 = (1 - 2*c) * (pi_0(p, k, r) + a * pi_1(p, k, r) - b * pi_2(p, k, r)) 
        term2 = c * (1 + a - b)
        return term_1 + term2

# Example usage:
r_value = 4
a_value = 0.1
b_value = 0.1
c_value = 0.1
k_value = 0.5
p_value = 0.25


for j in range(100):
    result = generalized_update(r_value, a_value, b_value, c_value, k_value, p_value)
    p_value = result
    print(result)

result = generalized_update(r_value, a_value, b_value, c_value, k_value, 0.9718)


def filter_close_numbers(numbers, tolerance=1e-1):
    unique_numbers = [numbers[0]]

    for num in numbers[1:]:
        if all(abs(num - existing_num) > tolerance for existing_num in unique_numbers):
            unique_numbers.append(num)

    return unique_numbers


def generalized_fixed_points(r_value, a_value, b_value, c_value, k_value, step_size=0.000001, tolerance=1e-3):
    fixed_points = []
    for x in range(int(1/step_size)):
        if x == 1 or x == 0:
            continue
        current_value = x * step_size
        next_value = generalized_update(r_value, a_value, b_value, c_value, k_value, current_value)

        if abs(next_value - current_value) < tolerance:
            fixed_points.append(current_value)
    return filter_close_numbers(fixed_points)

points = generalized_fixed_points(r_value, a_value, b_value, c_value, k_value)


# Update for a 

r_value = 7
a_values = [0.14, 0.24, 0.34]
b_value = 0
c_value = 0
k_value = 0.5

p_values = np.linspace(0.001, 1, 1000)

for a_value in a_values:
    updated_values = [generalized_update(r_value, a_value, b_value, c_value, k_value, p) for p in p_values]
    plt.plot(p_values, updated_values, label=f'a = {a_value}')

plt.plot(p_values, p_values, label='y = x', linestyle='--', color='red')

plt.xlabel('Previous Probability')
plt.ylabel('Current Probability')
plt.legend()
plt.title('Probability Transition of $P_{a,0,0,-}^{(7)}$ for different values of $a$.')
plt.grid(True)
plt.show()

# Update for a  and b

r_value = 7
a_values = [0.22, 0.32, 0.42]
b_value = 0.34
c_value = 0
k_value = 0.5

p_values = np.linspace(0.001, 0.99999, 1000)

for a_value in a_values:
    updated_values = [generalized_update(r_value, a_value, b_value, c_value, k_value, p) for p in p_values]
    plt.plot(p_values, updated_values, label=f'a = {a_value}')

plt.plot(p_values, p_values, label='y = x', linestyle='--', color='red')

plt.xlabel('Previous Probability')
plt.ylabel('Current Probability')
plt.legend()
plt.title('Probability Transition of $P_{a,0.32,0,-}^{(7)}$ for different values of $a$ and $b= 0.34$.')
plt.grid(True)
plt.show()


# Time dependent based on a


r_value = 3
a_values = [0.17, 0.172]
b_value = 0
c_value = 0
k_value = 0.5

for a_value in a_values:
    p_values = [0.25]
    time = [0]
    for j in range(200):
        time.append(time[-1] + 1)
        p_values.append(generalized_update(r_value, a_value, b_value, c_value, k_value, p_values[-1]))
        print(f'a={a_value}, time={time[-1]}, p={p_values[-1]}')
    plt.plot(time, p_values, label=f'a = {a_value}')

plt.xlabel('Time')
plt.ylabel('Transition Probability')
plt.legend()
plt.title('Evolution of $P_{a,0,0,-}^{(3)}$ with respect to time for different values of $a$.')
plt.grid(True)
plt.show()



# Time dependent based on c

r_value = 4
a_value = 0.1
b_value = 0.1
c_values = [0.04, 0.05]
k_value = 0.6

for c_value in c_values:
    p_values = [0.2]
    time = [0]
    for j in range(200):
        time.append(time[-1] + 1)
        p_values.append(generalized_update(r_value, a_value, b_value, c_value, k_value, p_values[-1]))
        print(f'c={c_value}, time={time[-1]}, p={p_values[-1]}')
    plt.plot(time, p_values, label=f'c = {c_value}')

plt.xlabel('Time')
plt.ylabel('Transition Probability')
plt.legend()
plt.title('Evolution of $P_{0.1,0.1,c,0.6}^{(4)}$ with respect to time for different values of $c$.')
plt.grid(True)
plt.show()
