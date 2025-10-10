(* ::Package:: *)

(*
   Mathematica helper utilities for analysing exported hillclimb time/gain series.

   Usage
   -----
   1. Export the canonical cumulative gain series:

        python hc_curve.py export-series activity.fit -o timeseries.csv

      or with the Rust CLI:

        cargo run -p hc_curve_cli -- export-series activity.fit -o timeseries.csv

   2. In Mathematica (14.3+), load this file and create a dataset:

        Get["/path/to/mathematica_gain_time.wl"];
        data = LoadTimeseries["/path/to/timeseries.csv"];

   3. Compute minimum-time windows for target gains:

        MinTimeReport[data, {50, 100, 150, 200}]

      The result is a list of associations with keys:
        "Gain" (meters), "MinTime" (seconds), "Start"/"End" (seconds from activity start),
        and "Rate" (meters per hour). Missing targets return `Missing` entries.
*)

ClearAll[LoadTimeseries, MinTimeWindow, MinTimeReport, PlotGainTimeSeries, PlotGainTimeCurve, CompareWithCsv, GainTimeSelfTest];

LoadTimeseries::usage = "LoadTimeseries[path] imports the exported CSV and returns an ordered list of {time_s, cumulative_gain_m}.";
MinTimeWindow::usage = "MinTimeWindow[data, gain] computes the minimal-duration window achieving at least `gain` meters of ascent.";
MinTimeReport::usage = "MinTimeReport[data, gains] computes MinTimeWindow for each target in gains.";
PlotGainTimeSeries::usage = "PlotGainTimeSeries[data, opts] renders the cumulative gain series (gain vs time).";
PlotGainTimeCurve::usage = "PlotGainTimeCurve[data, targets, opts] plots the gain-centric curve (gain vs min time in minutes).";
CompareWithCsv::usage = "CompareWithCsv[data, csv] compares Mathematica MinTimeReport results with the CLI gain_time.csv and reports deltas.";
GainTimeSelfTest::usage = "GainTimeSelfTest[] returns VerificationTest objects for the helper functions using synthetic data.";

LoadTimeseries[path_String?FileExistsQ] := Module[{raw},
  raw = Import[path, {"CSV", "Data"}];
  Rest[raw] /. {time_, gain_, ___} :> {N[time], N[gain]}
]

LoadTimeseries[path_String] := (Message[LoadTimeseries::nofile, path]; $Failed);
LoadTimeseries::nofile = "Timeseries CSV `1` could not be found.";

MinTimeWindow[data_List?MatrixQ, gain_?NumericQ] := Module[
  {times, gains, n, left = 1, best = Infinity, bestPair = None, target = N[gain], duration},
  times = data[[All, 1]];
  gains = data[[All, 2]];
  n = Length[times];

  If[target <= 0,
    Return[<|"Gain" -> target, "MinTime" -> 0., "Start" -> 0., "End" -> 0., "Rate" -> 0.|>]
  ];

  If[target > Last[gains] + 10^-9,
    Return[<|"Gain" -> target, "MinTime" -> Missing["NotAvailable"], "Start" -> Missing[], "End" -> Missing[], "Rate" -> Missing[]|>]
  ];

  Do[
    While[left <= right && gains[[right]] - gains[[left]] >= target,
      duration = times[[right]] - times[[left]];
      If[duration < best,
        best = duration;
        bestPair = {times[[left]], times[[right]]};
      ];
      left++;
    ],
    {right, 1, n}
  ];

  If[best === Infinity,
    <|"Gain" -> target, "MinTime" -> Missing["NotAvailable"], "Start" -> Missing[], "End" -> Missing[], "Rate" -> Missing[]|>,
    <|"Gain" -> target, "MinTime" -> best, "Start" -> First[bestPair], "End" -> Last[bestPair], "Rate" -> target/best*3600.|>
  ]
]

MinTimeReport[data_List?MatrixQ, gains_List] := MinTimeWindow[data, #] & /@ gains;

PlotGainTimeSeries[data_List?MatrixQ, opts : OptionsPattern[ListLinePlot]] := Module[
  {times, gains},
  times = data[[All, 1]]/60.;
  gains = data[[All, 2]];
  ListLinePlot[
    Transpose[{times, gains}],
    Frame -> True,
    FrameLabel -> {"Time (min)", "Cumulative gain (m)"},
    PlotTheme -> "Scientific",
    opts
  ]
];

Options[PlotGainTimeCurve] = {"Targets" -> {50, 100, 150, 200, 300, 500, 750, 1000}};
PlotGainTimeCurve[data_List?MatrixQ, opts : OptionsPattern[]] := Module[
  {targets, report, pts},
  targets = OptionValue["Targets"];
  If[targets === Automatic, targets = {50, 100, 150, 200, 300, 500, 750, 1000}];
  report = MinTimeReport[data, targets];
  pts = Cases[report, assoc_Association /; NumericQ[assoc["MinTime"]] :> {assoc["Gain"], assoc["MinTime"]/60.}];
  ListLinePlot[
    pts,
    Frame -> True,
    FrameLabel -> {"Gain (m)", "Minimum time (min)"},
    PlotMarkers -> Automatic,
    PlotTheme -> "Scientific",
    opts
  ]
];

Options[CompareWithCsv] = {"Tolerance" -> 1.0};
CompareWithCsv[data_List?MatrixQ, path_String?FileExistsQ, opts : OptionsPattern[]] := Module[
  {raw, header, rows, idxGain, idxTime, idxRate, idxNote, numeric, targets, cliTimes, cliRates, notes, report, rowsOut, tolerance, maxDelta},
  raw = Import[path, {"CSV", "Data"}];
  If[Length[raw] < 2, Return[<||>]];
  header = First[raw];
  rows = Rest[raw];
  idxGain = FirstPosition[header, "gain_m"];
  idxTime = FirstPosition[header, "min_time_s"];
  idxRate = FirstPosition[header, "avg_rate_m_per_hr"];
  idxNote = FirstPosition[header, "note"];
  If[!ListQ[idxGain] || !ListQ[idxTime] || !ListQ[idxRate],
    Message[CompareWithCsv::badheader, path];
    Return[$Failed];
  ];
  idxGain = idxGain[[1]];
  idxTime = idxTime[[1]];
  idxRate = idxRate[[1]];
  numeric[val_] := Which[
    NumericQ[val], N[val],
    StringQ[val], Quiet @ Check[N @ ToExpression[val], Missing[]],
    True, Missing[]
  ];
  targets = numeric /@ rows[[All, idxGain]];
  cliTimes = numeric /@ rows[[All, idxTime]];
  cliRates = numeric /@ rows[[All, idxRate]];
  notes = If[ListQ[idxNote], rows[[All, idxNote[[1]]]], ConstantArray[Missing[], Length[rows]]];
  validPos = Flatten @ Position[targets, _?NumericQ];
  targets = targets[[validPos]];
  cliTimes = cliTimes[[validPos]];
  cliRates = cliRates[[validPos]];
  notes = notes[[validPos]];
  If[Length[targets] == 0,
    Return[<|"Rows" -> {}, "MaxDeltaTime" -> 0., "WithinTolerance" -> True|>]
  ];
  report = MinTimeReport[data, targets];
  rowsOut = Table[
    Module[{wl = report[[i]], gain = targets[[i]], cliT = cliTimes[[i]], cliR = cliRates[[i]], note = notes[[i]], wlT, wlR, delta},
      wlT = Lookup[wl, "MinTime", Missing[]];
      wlR = Lookup[wl, "Rate", Missing[]];
      delta = If[NumericQ[wlT] && NumericQ[cliT], cliT - wlT, Missing[]];
      <|
        "Gain" -> gain,
        "CLI_MinTime" -> cliT,
        "WL_MinTime" -> wlT,
        "Delta_Time" -> delta,
        "CLI_Rate" -> cliR,
        "WL_Rate" -> wlR,
        "Note" -> note
      |>
    ],
    {i, Length[targets]}
  ];
  tolerance = OptionValue["Tolerance"];
  maxDelta = Module[{vals = DeleteMissing[Abs @ Lookup[rowsOut, "Delta_Time"]]}, If[Length[vals] == 0, 0., Max[vals]]];
  <|
    "Rows" -> rowsOut,
    "MaxDeltaTime" -> maxDelta,
    "WithinTolerance" -> maxDelta <= tolerance
  |>
];

CompareWithCsv[path_String?(!FileExistsQ[#] & ), ___] := (Message[CompareWithCsv::nofile, path]; $Failed);
CompareWithCsv::nofile = "Gain-time CSV `1` could not be found.";
CompareWithCsv::badheader = "Gain-time CSV `1` is missing required columns (gain_m/min_time_s/avg_rate_m_per_hr).";

GainTimeSelfTest[] := Module[
  {rate = 0.25, data, report, tests},
  data = Table[{t, rate t}, {t, 0, 600, 20}];
  report = MinTimeReport[data, {50, 150}];
  tests = {
    VerificationTest[Lookup[report[[1]], "MinTime"], 200., SameTest -> (Abs[#1 - #2] < 10^-6 &)],
    VerificationTest[Lookup[report[[2]], "Rate"], rate*3600., SameTest -> (Abs[#1 - #2] < 10^-6 &)]
  };
  tests
];
