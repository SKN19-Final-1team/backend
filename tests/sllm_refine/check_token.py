"""
PyKomoran Token 객체 구조 확인
"""

from PyKomoran import Komoran

k = Komoran("STABLE")
result = k.pos("테디카드로 결제")

print("Result type:", type(result))
print("Result:", result)

if result:
    print("\nFirst item:")
    print("  Type:", type(result[0]))
    print("  Value:", result[0])
    print("  Dir:", [attr for attr in dir(result[0]) if not attr.startswith('_')])
    
    # Token 객체 속성 확인
    first_token = result[0]
    if hasattr(first_token, 'morph'):
        print("  morph:", first_token.morph)
    if hasattr(first_token, 'pos'):
        print("  pos:", first_token.pos)
    if hasattr(first_token, 'first'):
        print("  first:", first_token.first)
    if hasattr(first_token, 'second'):
        print("  second:", first_token.second)
