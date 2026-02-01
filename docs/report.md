---
title: "Ubrzavanje algoritma za praćenje zraka u računarskoj grafici"
author: "Nikola Kušlaković E2 121/2025"
abstract: "U ovom radu se razmatra ubrzanje algoritma za praćenje zraka koristeći CUDA i OpenMP."
date: "30.01.2026."
toc: true
toc-title: "Sadrzaj"
bibliography: references.bib
link-citations: true
geometry: "margin=1in"
papersize: a4
header-includes: |
  \renewcommand{\abstractname}{Sažetak}
  \renewcommand{\figurename}{Slika}
  \renewcommand{\tablename}{Tabela}
  \usepackage{setspace}
  \onehalfspacing
  \usepackage{xcolor}
  \usepackage{mdframed}
  \usepackage{caption}
  \definecolor{bgcolor}{RGB}{240,240,240}
  \BeforeBeginEnvironment{Highlighting}{\begin{mdframed}[backgroundcolor=bgcolor, linewidth=1pt, innerleftmargin=5pt, innerrightmargin=5pt, innertopmargin=5pt, innerbottommargin=5pt]}
  \AfterEndEnvironment{Highlighting}{\end{mdframed}}
  \makeatletter
  \def\@maketitle{%
    \newpage
    \null
    \vspace*{\fill}
    \begin{center}%
    \let \footnote \thanks
      {\LARGE Fakultet tehničkih nauka}%
      \vskip 0.25em%
      {\normalsize Univerzitet u Novom Sadu}%
      \vskip 2em%
      {\normalsize Računarski sistemi visokih performansi}%
      \vskip 2em%
      {\huge \bfseries \@title \par}%
      \vskip 10.0em%
      {\large
        \lineskip .5em%
        \begin{tabular}[t]{c}%
          \@author
        \end{tabular}\par}%
      \vskip 1em%
      {\large \@date}%
    \end{center}%
    \vspace*{\fill}
    \par
    \newpage}
  \makeatother
  \let\oldtoc\tableofcontents
  \renewcommand{\tableofcontents}{\newpage\oldtoc\newpage}
---

\newpage

# Uvod

## Opis problema

Praćenje zraka (eng. *ray tracing*) je tehnika renderovanja u računarskoj grafici koja simulira fizičko ponašanje svetlosti. Umesto da projektuje poligone na ekran (kao kod rasterizacije), ovaj algoritam prati putanju pojedinačnih zraka svetlosti od kamere (oka posmatrača) kroz svaki piksel na ekranu, simulirajući njihovu interakciju sa objektima u sceni (refleksiju, refrakciju, senke itd.). U literaturi termin *ray tracing* se često koristi za bilo koju tehniku koja simulira fizičko ponašanje svetlosti, a ne samo onu koja prati putanju zraka.

U istoriji razvoja ove tehnike, ključan je Whitted-style ray tracing [@raytracing1980], koji je prvi omogućio realistične refleksije i prelamanja svetlosti. Međutim, on je bio deterministički i nije mogao verno da prikaže meke senke, indirektno osvetljenje ili ambijentalno okluziju.

U ovom radu, fokus je bio na metodi praćenja putanje zraka (eng. *path tracing*), modernijoj varijanti koja potpada pod Monte Carlo algoritme za renderovanje. Za razliku od Whitted-ovog modela, Path tracing koristi nasumično uzorkovanje zrakova kako bi rešio jednačinu renderovanja. On prati putanje zraka koji se nasumično odbijaju od površina, čime se postiže fotorealizam i prirodno globalno osvetljenje (eng. *global illumination*).

Ray tracing se danas najčešće koristi u:

- Filmskoj industriji i animaciji - Za postizanje vrhunskog fotorealizma gde vreme renderovanja nije kritično (offline rendering).
- Video igrama - Iako su grafičke kartice danas jako performantne, ray tracing je još uvijek relativno spor, pa je zbog toga danas korišćena mnoštvo optimizacija kako bi se postigla optimalna performansa. Najčešće se koriste hibridne metode renderovanja koje kombiniraju ray tracing i rasterizaciju kako bi se postigla optimalna performansa.

- Arhitektonskoj vizuelizaciji i 3D modelovanju - Za precizan prikaz materijala i svetlosti u prostoru, kao i za modeliranje i vizuelizaciju prostora.

Glavni razlog zašto je ray tracing spor je ogromna količina računanja. Da bi se dobila čista slika bez šuma u path tracing-u, potrebno je ispaliti stotine ili hiljade zraka po svakom pikselu. Svaki zrak mora da proveri koliziju sa hiljadama (ili milionima) trouglova u sceni. Čak i uz optimizacije poput bolje pretrage prostora (Bounding Volume Hierarchy), proces zahteva masivnu procesorsku snagu, zbog čega je idealan kandidat za ubrzanje. Ovaj problem se može svrstati i u smešno jednostavne probleme [@embarrassingly_parallel] za ubrzanje na grafičkoj kartici, jer je praćenje jednog zraka nezavisno od ostalih zrakova.

## Cilj

Cilj ovog rada je implementacija path tracing-a na CPU i GPU hardveru koristeći CUDA i OpenMP radne okvire, kao i poređenje performansi između CPU i GPU implementacije. Osim toga, fokus je bio i na dizajnu softverskog rešenja gde je u obzir uzeta prenosivost kako bi se postigla maksimalna fleksibilnost i olakšana podrška za buduću nadogradnju.

\newpage

# Implementacija

## Korišćene tehnologije i hardver

Hardver koji je korišćen za testiranje je:

- CPU: AMD Ryzen 7 7700, 8 jezgara, 16 niti
- GPU: NVIDIA GeForce RTX 3070, 8 GB VRAM
- RAM: 32 GB
- OS: Windows 11, pokretano u WSL2 Ubuntu 22.04.05 LTS

Tehnologije korišćene za razvoj su:

- CUDA 11.5
- OpenMP 4.5
- GCC 11.4.0
- CMake 3.22.1
- GLM 0.9.9.8

GLM je biblioteka za matematiku koja je korišćena za vektorske i matrične operacije. Često je korišćena u programiranju grafike i fizike, jer implementira tipove kao što su `vec3`, `vec4`, `mat4` itd.

## Struktura projekta

Projekat je organizovan u tri glavna modula:

- **CORE** modul - Osnovni paket koji sadrži sve ključne komponente algoritma za praćenje zraka (kamera, zraci, sfere, materijali, svet itd.). Glavni izazov implementacije bio je dizajnirati ovaj paket kao višeplatformski, tako da isti kod može da se prevodi i izvršava i na CPU i na GPU hardveru.

- **CPU** modul - Sekvencijalna CPU implementacija koja koristi OpenMP za paralelizaciju ili sekvencijalno izvršavanje. Ovaj modul integriše `CORE` paket i implementira glavni program za CPU izvršavanje.

- **CUDA** modul - Paralelna GPU implementacija koristeći CUDA. Ovaj modul takođe koristi `core` paket, ali ga prevodi kao CUDA kod i izvršava na grafičkoj kartici.

## Dizajn CORE modula

Pošto je bilo potrebno napraviti CORE modul višeplatformskim, bilo je potrebno pisati metode i klase na pametan način tako da one mogu da se prevedu i izvršavaju na i na procesoru i na grafičkoj kartici. Bitna ograničenja su bila seledeća:

1. Izbegavanje korišćenja rekurzije zbog relativno plitkog steka koji je dostupan nitima na grafičkoj kartici
2. Izbegavanje korišćenja polimorfizma i virtualnih metoda jer ovo najčešće nije dovoljno dobro podržano i optimizovano da se izvršava na grafičkoj kartici
3. Izbegavanje korišćenja `std::vector`, `std::map` i `std::shared_ptr` sličnih kolekcija iz standardne C++ biblioteke pošto ih standardna biblioteka CUDA ne implementira

Ograničenje 1. je prevaziđeno prevođenjem rekurzije u iterativno izvršavanje, dok je ograničenje 2. rešeno korišćenjem ugrađenog tipa `union`. Što se tiče ograničenja 3., korišćeni su čisti pokazivači na objekte i njihovi tipovi su bili fiksni i poznati u vremenu prevođenja.

GLM biblioteka je kompatibilna sa CUDA radnim okvirom, tako da je moguće prevesti je korišćenjem nvcc kompajlera (NVIDIA CUDA Compiler). Kompatibilnost sa CUDA okvirom urađen je kroz sledeću `hpp` datoteku unutar CORE modula:

\

```cpp
// cuda_compat.hpp
#pragma once

// CUDA support: include cuda.h first to define CUDA_VERSION for GLM
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>

// Define GLM settings for CUDA compatibility
#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif
#ifndef GLM_FORCE_PURE
#define GLM_FORCE_PURE
#endif

#else
// Non-CUDA builds: provide compatible macros
#define __device__
#define __host__
#define __global__
#define __forceinline__ inline
#endif

#include <glm/glm.hpp>
```

\captionof{lstlisting}{Code snippet 1: GLM CUDA compatibility header}


# Rezultati i diskusija

# Zaključak

# Literatura
