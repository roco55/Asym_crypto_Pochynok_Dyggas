# %% [markdown]
# # Лабораторна робота 2 з "Асиметричних криптосистем та протоколів"
# ## Тема: Вивчення криптосистеми RSA та алгоритму електронного підпису; ознайомлення з методами генерації параметрів для асиметричних криптосистем
# 
# **Виконали**\
# Дигас Богдан, ФІ-03\
# Починок Юрій, ФІ-03

# %%
import random
import math
import numpy as np
rand = random.SystemRandom()

# %% [markdown]
# ## Miller-Rabin test

# %%

def decomposing_number(n, a):
    exp = n - 1
    while not exp & 1:  # while exp is even
        exp >>= 1  # divide by 2
    if pow(a, exp, n) == 1:
        return True  # number is composite
    while exp < n - 1:
        if pow(a, exp, n) == n - 1:
            return True  # number is composite
        exp <<= 1  # multiply by 2
    return False  # number is probably prime


def miller_rabbin_test(n, k=20):
    for i in range(k):
        a = rand.randrange(1, n - 1)
        if not decomposing_number(n, a):
            return False  # number is composite
    return True  # number is probably prime

def bin_to_dec(bin_n):
    dec_n = 0
    res = 0
    for i in range(len(bin_n)):
        res = bin_n[len(bin_n) - i - 1] * 2 ** i
        dec_n += res
    return dec_n


# %% [markdown]
# ## Prime number generation

# %%
def generate_bit_seq(n):
    seq = [0]*n
    for i in range(n):
        seq[i] = rand.randint(0, 1)
    return seq


def L20(n):
    seq = generate_bit_seq(20)
    result = [0]*n
    for i in range(20):
        result[i] = seq[i]
    for i in range(20,n):
        result[i] = result[i-3]^result[i-5]^result[i-9]^result[i-20]
    return result


# %%
def generate_prime_number(x):
    res = [1,0,0]
    while(miller_rabbin_test(bin_to_dec(res)) == False):
        res = L20(x)
    return res

# %% [markdown]
# ## Better prime number generation 2*k*p+1

# %%
def generate_better_prime_numbers(n):
    p = bin_to_dec(generate_prime_number(n))
    q = bin_to_dec(generate_prime_number(n))
    
    i = 1
    while(miller_rabbin_test(2*p*i + 1) == False):
        i = i + 1   

    better_p = 2*p*i + 1
    
    j = 1
    while(miller_rabbin_test(2*q*j + 1) == False):
        j = j + 1

    better_q = 2*q*j + 1
    
    return better_p, better_q

# %% [markdown]
# ## Initialization 

# %%
def RSA_gen_constants():
    p,q = generate_better_prime_numbers(256)
    e = 2**16 + 1
    n = p*q
    phi_n = (p-1)*(q-1)
    d = pow(e,-1,phi_n)
    
    return p,q,n,e,d

# %% [markdown]
# ## User and his parameters

# %%
class User:
    
    __private_key = None
    __key_pair = None
    
    __k = None

    def __init__(self):
        result = RSA_gen_constants()
        self.__key_pair = result[0], result[1]
        self.public_n = result[2]
        self.public_e = result[3]
        self.__private_key = result[4]
        
    def show(self):
        print(self.__key_pair, self.public_n, self.public_e, self.__private_key, sep = "\n")
        
    
    def RSA_decrypt(self, C):
        res = pow(C, self.__private_key, self.public_n)
        return res
    
    def RSA_sign(self, M):
        res = pow(M, self.__private_key, self.public_n)
        return res
    
    def Send_Key(self, e_1, n_1):
        while(self.public_n > n_1):
            print("Regenerating keys")
            print(self.public_n," && ", n_1)
            result = RSA_gen_constants()
            self.__key_pair = result[0], result[1]
            self.public_n = result[2]
            self.public_e = result[3]
            self.__private_key = result[4]
        
        k = random.randint(1, self.public_n-1)
        print("initial k, ", hex(k))
        S = pow(k, self.__private_key, self.public_n)
        S_1 = pow(S, e_1, n_1)
        k_1 = pow(k, e_1, n_1)
        
        return k_1, S_1
    
    def Receive_Key(self, k_1, S_1, e, n):
        k = pow(k_1, self.__private_key, self.public_n)
        S = pow(S_1, self.__private_key, self.public_n)
        
        if k == pow(S, e, n):
            __k = k
            print("k is authentic, ", hex(k))
        else:
            print("k is not authentic, try again")

# %% [markdown]
# ## User initialization

# %%
A = User()
B = User()

def RSA_encrypt(m,e,n):
    res = pow(m,e,n)
    return res

def RSA_verify(S, M, e, n):
    res = pow(S,e,n)
    return res == M

# %%
print("Public n: ", A.public_n)
print("Public e: ", A.public_e)

# %% [markdown]
# ## Next corresponding functions are working with http://asymcryptwebservice.appspot.com/?section=rsa website and were created to check their corectness

# %% [markdown]
# ## Server Key

# %%
server_key = int("8C9B7B4ADF8D45CF4CB63183681109D2D38B0941F6C04971FD3C0BC55CD99647", 16)
server_e = 2**16+1

# %% [markdown]
# ## Encryption

# %%
M = int("12345", 16)
print("Public n: ",hex(A.public_n))
print("Public e: ",hex(A.public_e))
print("Decrypted M: ",hex(RSA_encrypt(M,server_e,server_key)))

# %% [markdown]
# ## Decryption

# %%
C = int("028CA31431ED51576FFA245A3E3A044287233FDFD2C292D15FA7FDAEC4F001D0BDE8B684084CA914FE6B878A01C00DE0AC4AC1E227982C91D5B7E62036D49D5C9106", 16)
print("Encrypted M: ", hex(A.RSA_decrypt(C)))

# %% [markdown]
# ## Signature

# %%
M = int("12345", 16)
server_S = int("36880956AD2EDC33F2C8B731AA6823556FDE2F007A261824908F61144D10D1FC",16)
print("Verification (True/False): ",RSA_verify(server_S, M, server_e, server_key))

# %% [markdown]
# ## Verification

# %%
M = int("1234", 16)
print(hex(A.RSA_sign(M)))
print(hex(A.public_n))
print(hex(A.public_e))
# result is presented on the website

# %% [markdown]
# ## Send key
# 

# %%
print(hex(A.public_n))
print(hex(A.public_e))

# %%
server_encrypted_key = int("01D46C75E2CF703ACB160AE5615901A31409C708BB76E09FF4FF6E29EEBA497F6C2DE29E8D3F7D8193967DDDEACB0764F18D044E40C1E44C16DFEB5CE36C9CF2CF39", 16)
server_encrypted_signature = int("488AF41BAB8BD994803C9557808BD8723A9CB4E46109D15B88D8D05A83460EFABB5554453C0BD5C4E6E9AB440C00F2181ABF770664578965A99EEF5DF98C0D1F37",16)
A.Receive_Key(server_encrypted_key,server_encrypted_signature, A.public_e, server_key)

# %% [markdown]
# ## Recieve Key

# %% [markdown]
# ### для роботи даної функції потрібно збільшити модуль сервера з яким ми спілкуємось, а бо ж зменшити ключ абонента, що робиться трохи вище

# %%
server_key = int("87AE7AEB5B6DB02F738224DC2CB927132FC7E0D2ED094EFBCE540B4622053BE77E36B33C72904CE80E5493607681BF6EF7DBE5A3994B045059CD81DBF7EEA2D82AC260E6B1840CCFCD429850EA453BEC595A7B74DBFFD65F26EAF7EFCA4DA6238F80AFE8D2F2372523B44C1F1F97505FA3D0ACD1715DD5BCEB45C475B9FDF693", 16)
server_e = 2**16+1
k_1, s_1 = A.Send_Key(server_e, server_key)
print(hex(k_1))
print(hex(s_1))
print(hex(A.public_n))
print(hex(A.public_e))
# result is presented on the website

# %% [markdown]
# ## Conclusion

# %% [markdown]
# По ходу виконання роботи не виникало значних труднощів.
# 
# Опрацювання алгоритмів і логіки працювання RSA були цікавими в реалізації й на невеликих числах, як виявилось достатньо швидким.
# 
# Алгоритм устворення *кращих* простих чисел виявився досить непердбачуваним так як $k$ у формулі $2kp+1$ могло досягати великих значень й збільшувати нове просте число майже вдвічі. (за кількістю бітів)
# 
# Лабораторна робота була не складною але цікавою, RSA ефективним і повідомлення захищеними :)


