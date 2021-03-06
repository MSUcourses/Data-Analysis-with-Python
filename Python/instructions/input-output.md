# Рекомендации по работе с вводом-выводом

Первым делом, при чтении домашних задач нужно обращать внимание на таблицу ограничений к решению задания. Она указывается в самом начале задания. В ней отображается информация об ограничении по времени работы, по памяти, а также ограничения на ввод и вывод.

<img width="600" alt="copy on disk" src="https://user-images.githubusercontent.com/28728575/154576708-874e7f37-5019-4598-931a-504c66c00f84.png">


Ограничения по времени и по памяти – как правило, не сильно отличаются от задачи к задаче и нужны в первую очередь для корректной работы тестовой системы. Как правило, они выбираются с некоторым запасом, поэтому можно считать, что ваша программа практические всегда будет удовлетворять этим ограничениям.

Более важно при работе с текстом условия обращать внимание на строки «Ввод» и «Вывод». В них содержится информация о том, как ваша программа будет взаимодействовать с тестовой системой.

Практически любая программа, решаемая в рамках домашних заданий, представляет собой некоторый «чёрный ящик», который каким-то образом (в соответствии с написанным вами кодом) преобразует некоторые входные данные в выходные.

Например, программа, которая вычисляет сумму двух чисел, на вход должна получить 2 числа, а на выходе – вернуть одно число (сумму входных чисел).

В случае с домашними задачами из контеста, входные данные программа должна получать из тестовой системы, а также возвращать их туда же.
Поэтому, давайте обсудим, как корректно работать со входными и выходными данными, исходя из условия задачи.

## Работа со вводом

Как правило, допускаются 4 способа передачи входных данных из тестовой системы:
1. Отсутствует. \
Это значение указывается в задачах, где не нужно ничего получать из тестовой системы (Например, в условии уже указывается строка, с которой нужно работать).
В этом случае, не нужно считывать информацию в программе.
2. Стандартный ввод. \
В этом случае, программа должна включать в себя ввод исходных данных с клавиатуры посредством input() (об этом – смотри ниже).
3. `input.txt` (Или «Файл»).\
В этом случае, программа должна получать исходные данные из файла (посредством `open()`, `readlines()`, `close()` и т.д.)
4. Комбинированный ввод (стандартный ввод или `input.txt`).
В этом случае можно использовать ЛИБО `input()`, ЛИБО работу с файлами (что именно использовать – решаете вы сами, по желанию. Оба варианта подходят).

Давайте рассмотрим эти методы поподробнее.

### Стандартный ввод – ввод с клавиатуры.

Такая формулировка ограничения на ввод говорит нам о том, что исходные данные в программе должны вводиться С КЛАВИАТУРЫ.
Т.е. в вашей программе должен присутствовать `input()`.

Например, в задаче: «Вывести первый символ из строки, введённой с клавиатуры,» - нужно будет считать  данную строку с помощью `input()`:
```python
S = input()
```

При этом, стоит отметить, что в параметрах `input()` можно указать строку-подсказку, которая выведется на экран перед вводом соответствующего значения с клавиатуры.
Тестовая система рассматривает такую строку-подсказку как сообщение, выводимое на стандартный вывод (см. ниже) и не засчитывает такое решение.
Поэтому, при работе со стандартным вводом в тестовой системе **нужно использовать `input()` без параметров**!

### Ввод чисел с клавиатуры

В ряде задач подразумевается ввод чисел с клавиатуры. Однако, `input()` подразумевает ввод __строк__. Поэтому, в таких задачах нужно явно сообщить системе, что введённая с клавиатуры строка представляет собой число. Для этого можно использовать __`int()`__ (если введённое число - целое) или __`float()`__ (если введённое число - вещественное).

Например, запись: `a = input()` говорит, что `a` – это строка.
А запись:  `a = int(input())` - говорит нам о том, что мы должны считывать с клавиатуры целое число.

### Ввод чисел из файла `input.txt`

В данном случае подразумевается, что исходные для задачи данные нужно считывать из файла с указанным именем (обычно это `input.txt`).

Это делается посредством функций работы с файлами: `open()`, `readlines()`, `close()` и т.д.:
```python
f = open(‘input.txt’)
lines = f.readlines()
f.close()
```

Обратите внимание, что в open нужно указывать корректное имя входного файла (оно указывается в условии задачи).

Кроме того, не нужно забывать закрывать файл (с помощью `close()`).

## Работа с выводом

Здесь важно понимать, что работа с выводом в _Google Collab_ и в тестовой системе немного отличается.

Например, чтобы вывести значение переменной `var` в Google Collab достаточно просто написать имя этой переменной.


<img width="600" alt="copy on disk" src="https://user-images.githubusercontent.com/28728575/154578859-086ce945-d3ee-404b-a56c-2a8807529967.png">

__Для вывода же результатов работы программы в тестовую систему необходимо явно прописывать соответственную команду вывода (например, `print()`).__

По аналогии с вводом, допускаются 3 способа передачи выходных данных в тестовую систему:
1. Стандартный вывод (или вывод на экран).\
В этом случае данные должны выводиться на экран посредством функции `print()`.
2.	`output.txt` (Или «Файл»). \
В этом случае, программа должна получать исходные данные из файла (посредством `open()`, `writelines()`, `close()` и т.д.).
В данном случае, работа с файлами выполняется аналогично считыванию данных из файла (см. выше), только для открытия файла нужно явно указывать параметр `w`: `open(‘output.txt’, ‘w’)`
3.	Комбинированный ввод (стандартный вывод или `output.txt`). \
В этом случае можно использовать ЛИБО `print()`, ЛИБО работу с файлами (что именно использовать – решаете вы сами, по желанию. Оба варианта подходят).









