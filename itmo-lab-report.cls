\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{itmo-lab-report}

\LoadClass[12pt]{article}

\RequirePackage[utf8]{inputenc}
\RequirePackage[russian]{babel}
\RequirePackage[
    a4paper,
    total={170mm,257mm},
    left=30mm,
    right=10mm,
    top=20mm,
    bottom=20mm]{geometry}
\RequirePackage{booktabs}
\RequirePackage{amsmath}
\RequirePackage{amssymb}

\newcommand*{\serialnumber}[1]{\gdef\@serialnumber{#1}}
\renewcommand*{\title}[1]{\gdef\@title{#1}}
\newcommand*{\studentname}[1]{\gdef\@studentname{#1}}
\newcommand*{\studentgroup}[1]{\gdef\@studentgroup{#1}}
\renewcommand*{\maketitle}{%
    \begin{titlepage}
        \centering
        {\footnotesize МИНИСТЕРСТВО~НАУКИ~И~ВЫСШЕГО~ОБРАЗОВАНИЯ РОССИЙСКОЙ~ФЕДЕРАЦИИ \par}
        \vspace{0.25cm}
        {\footnotesize\bfseries ФЕДЕРАЛЬНОЕ~ГОСУДАРСТВЕННОЕ~АВТОНОМНОЕ ОБРАЗОВАТЕЛЬНОЕ~УЧРЕЖДЕНИЕ~ВЫСШЕГО~ОБРАЗОВАНИЯ
        ``НАЦИОНАЛЬНЫЙ~ИССЛЕДОВАТЕЛЬСКИЙ~УНИВЕРСИТЕТ~ИТМО'' \par}
        \vspace{0.5cm} \hrule \vspace{0.6cm}
        { Факультет программной инженерии и компьютерной техники \par}
        \vspace{7cm}
        {\large\bfseries Лабораторная работа №\@serialnumber \par}
        {\large по дисциплине ``Вычислительная математика'' \par}
        {\large ``\@title'' \par}
        \vspace{2cm}
        {\raggedleft Выполнил: \par студент группы \@studentgroup\par \@studentname\unskip\strut\par}
        \vspace{1cm}
        {\raggedleft Преподаватель: \par Перл О.В. \par}
        \vfill
        { Санкт-Петербург \par}
        { 2022 \par}
    \end{titlepage}
}

\RequirePackage[hidelinks]{hyperref}
\RequirePackage{titlesec}
\titlespacing*{\section}{0pt}{10ex}{4ex}
%\setcounter{secnumdepth}{0}

\RequirePackage{tikz}
\usetikzlibrary{shapes.geometric, arrows}
\tikzset{
    terminal/.style={
        rectangle, rounded corners, draw=black,
        minimum width=3cm, minimum height=1cm
    },
    io/.style={
        trapezium, trapezium left angle=70, trapezium right angle=110, draw=black,
        trapezium stretches=true, minimum width=3cm, minimum height=1cm
    },
    process/.style = {
        rectangle, draw=black, minimum width=3cm, minimum height=1cm
    },
    decision/.style = {diamond, aspect=3, minimum width=3cm, minimum height=1cm, text centered, draw=black},
    arrow/.style = {thick,->,>=stealth},
}

\RequirePackage[outputdir=./out]{minted}
\newmintedfile[pythoncode]{python}{breaklines,fontsize=\small}

\newenvironment{examples}[3]{
    \begin{table*}[ht]
        \centering \ttfamily \footnotesize
        \begin{tabular}{p{#1\textwidth}p{#2\textwidth}p{#3\textwidth}}
            \toprule Problem & Answer & Output \\
            }{
            \bottomrule
        \end{tabular}
    \end{table*}
}
\newcommand{\example}[3]{\midrule #1 & #2 & #3 \\}
\newcommand{\n}{\newline}
