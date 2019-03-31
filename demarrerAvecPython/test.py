quotes=["jean","maurice","philippe"]

def get_random_quote(my_list):
  return my_list[0]

print(get_random_quote(quotes))

user_decision=input("Tapez B pour quitter et une autre touche pour voir un autre proverbe: ")

while user_decision!="B":
  print(get_random_quote(quotes))
  user_decision=input("Tapez B pour quitter et une autre touche pour voir un autre proverbe: ")


