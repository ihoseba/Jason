from datetime import datetime, timedelta

fecha1 = datetime(2024, 10, 1)
fecha2 = datetime(2024, 10, 2)

if fecha1 < fecha2:
    print('fecha1 es menor que fecha2')

if fecha1 > fecha2:
    print('fecha1 es mayor que fecha2')

if fecha1 == fecha2:
    print('fecha1 es igual a fecha2')

diferecia = fecha2 - fecha1
print(diferecia)

fecha2 = fecha2 + timedelta(days=1)
fecha2_mas_2h = fecha2 + timedelta(hours=2)

print("fecha2_mas_2h", fecha2_mas_2h)
