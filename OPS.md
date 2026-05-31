# Отчет по семинару: Операторно-параметрические схемы имитационной модели

**Студент:** [Громов Владислав / ИУ5-63Б]  
**Тема:** Отрисовка ОПС модели процесса ремонта «Прибор-Мастер» в Mermaid

---

## 1. Трек 1 — Блок «ПРИБОР»

Операторно-параметрическая схема функционирования прибора, фиксирующая моменты поломки, занятость мастера и параметры состояния.

```mermaid
flowchart LR
    title[<em>Блок ПРИБОР</em>]
    
    %% Описание трека (операторы)
    h1(["BPEMЯ = Тслом"]) ==e100==> h2["COCT :=<br>сломан"]
    h2 ==e11==> h3(["режим =<br>работа"])
    h3 ==> h4(["Мастер<br>= своб"])
    h4 ==> h5["Мастер := занят<br>Трем := func(x)"]
    h5 ==e20==> h6(["BPEMЯ = Трем"])
    h6 ==> h7["сост := рабочий<br>мастер := своб<br>Тслом := func(x)"]
    h7 ==> h1

    %% Описание параметров
    par1((режим))
    par2((мастер))
    par3((COCT))
    par4((Трем))

    %% Связи с параметрами
    h2 -.-> par3
    h7 -.-> par3
    par1 -.-> h3
    par2 -.-> h4
    h5 -.-> par2
    h7 -.-> par2
    h5 -.-> par4
    par4 -.-> h6

    %% Инициализаторы
    Ini["I::"] .-> h1
    HTf["I:: Тслом = 100"] .-> h1

    %% Настройка стилей
    classDef cond fill:#bee, stroke:#aaa, stroke-width:1px;
    classDef state fill:#9e8, stroke:#333, stroke-width:1px;
    class h2,h5,h7 state;
    class h1,h3,h4,h6 cond;
    
    style title fill:yellow, stroke:red;
    style par1 fill:#fcc, stroke:#111, stroke-width:2px;
    style par2 fill:#fae, stroke:#bbb, stroke-width:2px;
    style par4 fill:#ccc, stroke:#555, stroke-width:2px;

    %% Стили видов связей
    linkStyle default stroke:red, stroke-width:4px;
    e11@{ curve: natural }
    e20@{ curve: stepAfter }
```

## 2. Трек 2 — Блок «МАСТЕР»

```mermaid
flowchart LR
    title2[<em>Блок МАСТЕР</em>]
    Ini2["I::"] .-> h11
    HTf2["Траб = 9ч"] .-> h11

    %% Трек Мастера
    h11["режим :=<br>отдых"] ==> h12(["BPEMЯ = Траб"])
    h12 ==> h13["режим := работа<br>Тотд := func"]
    h13 ==> h14(["BPEMЯ = Тотд"])
    
    %% Узел навигации (ромб)
    h15{"Мастер<br>= ..."}
    
    h14 ==> h15
    h15 ==>|"...= занят"| h16["Трем := Трем+<br>Траб - BPEMЯ"]
    h16 ==> h17(["Траб := func"])
    h17 ==> h18(["Траб > Траб - BPEMЯ"])
    h18 ==> h11
    
    h15 ==>|"...= своб"| h11

    %% Параметры
    parM1((режим))
    parM2((мастер))
    parM3((Трем))

    h11 -.-> parM1
    h13 -.-> parM1
    parM2 -.-> h15
    h16 -.-> parM3

    %% Настройка стилей
    classDef cond fill:#bee, stroke:#aaa, stroke-width:1px;
    classDef state fill:#9e8, stroke:#333, stroke-width:1px;
    classDef navig fill:#eda, stroke:#333, stroke-width:1px;
    
    class h12,h14,h17,h18 cond;
    class h11,h13,h16 state;
    class h15 navig;
    
    style title2 fill:yellow, stroke:red;
    style parM1 fill:#fcc, stroke:#111, stroke-width:2px;
    style parM2 fill:#fae, stroke:#bbb, stroke-width:2px;
    style parM3 fill:#ccc, stroke:#555, stroke-width:2px;
    
    linkStyle default stroke:black, stroke-width:2px;
```

## 3. Объединенная ОПС модели ремонта «Прибор-Мастер»

```mermaid
flowchart TD
    %% Глобальный стиль связей по умолчанию
    linkStyle default stroke:red, stroke-width:2px;

    %% --- ПОДГРАФ 1: ПРИБОР ---
    subgraph EQUIP [Блок ПРИБОР]
        direction TB
        h1([BPEMЯ = Тслом]) ==> h2[COCT := сломан]
        h2 ==> h3([режим = работа])
        h3 ==> h4([Мастер = своб])
        h4 ==> h5[Мастер := занят<br>Трем := func]
        h5 ==> h6([BPEMЯ = Трем])
        h6 ==> h7[сост := рабочий<br>мастер := своб]
        h7 ==> h1

        par_c((COCT))
        par_m((мастер))
        h2 -.-> par_c
        par_m -.-> h4
    end

    %% --- ПОДГРАФ 2: МАСТЕР ---
    subgraph MASTER [Блок ... МАСТЕР]
        direction TB
        h11([режим: отдых]) ==> h12([BPEMЯ = Траб])
        h12 ==> h13([режим := работа])
        h13 ==> h14([BPEMЯ = Тотд])
        h15{Мастер?}
        
        h14 ==> h15
        h15 -->|занят| h16[Трем := Трем + ...]
        h15 -->|своб| h11
        h16 ==> h11
        
        par_reg((режим))
        h11 -.-> par_reg
    end

    %% Межблочные связи через общие параметры
    par_m --- par_reg

    %% Применение классов стилей к объединенной схеме
    classDef state fill:#9e8, stroke:#333;
    class h2,h5,h7,h16 state;
    
    %% Настройка интерактивных ссылок по клику на параметры
    click par_m href "[https://iu5.bmstu.ru](https://iu5.bmstu.ru)" "Переход для Мастера" _blank
    click par_c href "[https://iu5.bmstu.ru](https://iu5.bmstu.ru)" "Параметр Состояние" _blank
```  