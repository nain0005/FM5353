using System;
using System.Collections.Generic;

class NormalGenerator
{
    private readonly Random _uniform;
    public NormalGenerator(int? seed = null)
    {
        _uniform = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    //  Standard Box-Muller: returns one Z ~ N(0,1)
    public double GenerateStandard()
    {
        double u1 = _uniform.NextDouble();
        double u2 = _uniform.NextDouble();
        if (u1 < 1e-15) u1 = 1e-15;                   
        return Math.Sqrt(-2.0 * Math.Log(u1))
               * Math.Cos(2.0 * Math.PI * u2);
    }

    public (double Z, double ZAntithetic) GenerateAntithetic()
    {
        double z = GenerateStandard();
        return (z, -z);
    }

    public static double VanDerCorput(int n, int b)
    {
        double result   = 0.0;
        double fraction = 1.0;

        while (n > 0)
        {
            fraction /= b;
            result   += (n % b) * fraction;
            n        /= b;
        }

        return result;
    }

    public static double VanDerCorputNormal(int index, int base1, int base2)
    {
        double u1 = VanDerCorput(index, base1);
        double u2 = VanDerCorput(index, base2);
        if (u1 < 1e-15) u1 = 1e-15;
        return Math.Sqrt(-2.0 * Math.Log(u1))
               * Math.Cos(2.0 * Math.PI * u2);
    }
}

class MarketData
{
    public double SpotPrice  { get; set; }   // S₀ — current stock price
    public double Volatility { get; set; }   // σ  — annualised volatility
    public double DividendYield { get; set; }// q  — continuous dividend yield

    public MarketData(double spot, double volatility, double dividendYield = 0.0)
    {
        if (spot       <= 0) throw new ArgumentException("Spot must be positive.");
        if (volatility <= 0) throw new ArgumentException("Volatility must be positive.");
        SpotPrice     = spot;
        Volatility    = volatility;
        DividendYield = dividendYield;
    }
}
class YieldCurve
{
    // Internal class: one point on the yield curve
    private class RatePillar
    {
        public double Tenor { get; }
        public double Rate  { get; }
        public RatePillar(double tenor, double rate) { Tenor = tenor; Rate = rate; }
    }

    private readonly List<RatePillar> _pillars = new List<RatePillar>();

    // Add a (tenor, rate) point and keep the list sorted by tenor
    public void AddRatePillar(double tenor, double rate)
    {
        _pillars.Add(new RatePillar(tenor, rate));
        _pillars.Sort((a, b) => a.Tenor.CompareTo(b.Tenor));
    }

    // Linear interpolation between pillars
    public double GetRate(double T)
    {
        if (_pillars.Count == 0)
            throw new InvalidOperationException("Yield curve has no pillars.");

        if (T <= _pillars[0].Tenor)
            return _pillars[0].Rate;

        if (T >= _pillars[_pillars.Count - 1].Tenor)
            return _pillars[_pillars.Count - 1].Rate;

        for (int i = 0; i < _pillars.Count - 1; i++)
        {
            double t0 = _pillars[i].Tenor,     r0 = _pillars[i].Rate;
            double t1 = _pillars[i + 1].Tenor, r1 = _pillars[i + 1].Rate;
            if (T >= t0 && T <= t1)
            {
                double alpha = (T - t0) / (t1 - t0);
                return r0 + alpha * (r1 - r0);
            }
        }

        return _pillars[_pillars.Count - 1].Rate;
    }

    // Martingale discount factor: P(0,T) = e^{-r(T)·T}
    public double DiscountFactor(double T) => Math.Exp(-GetRate(T) * T);
}

abstract class Option
{
    public double StrikePrice { get; }   // K — exercise price
    public double TimeToExpiry { get; }  // T — years to maturity

    protected Option(double strikePrice, double timeToExpiry)
    {
        if (strikePrice  <= 0) throw new ArgumentException("Strike must be positive.");
        if (timeToExpiry <= 0) throw new ArgumentException("Expiry must be positive.");
        StrikePrice  = strikePrice;
        TimeToExpiry = timeToExpiry;
    }

    public abstract double ComputePayoff(double terminalPrice);

    public abstract string Label { get; }
}

class EuropeanCall : Option
{
    public EuropeanCall(double strikePrice, double timeToExpiry)
        : base(strikePrice, timeToExpiry) { }

    // Call payoff: max(S_T − K, 0)
    public override double ComputePayoff(double terminalPrice)
        => Math.Max(terminalPrice - StrikePrice, 0.0);

    public override string Label
        => $"European Call   K = {StrikePrice:F2}   T = {TimeToExpiry:F2} yr";
}

class EuropeanPut : Option
{
    public EuropeanPut(double strikePrice, double timeToExpiry)
        : base(strikePrice, timeToExpiry) { }

    // Put payoff: max(K − S_T, 0)
    public override double ComputePayoff(double terminalPrice)
        => Math.Max(StrikePrice - terminalPrice, 0.0);

    public override string Label
        => $"European Put    K = {StrikePrice:F2}   T = {TimeToExpiry:F2} yr";
}
class PricingResult
{
    public double OptionPrice              { get; set; }
    public double StandardError           { get; set; }
    public double ConfidenceIntervalLow   => OptionPrice - 1.96 * StandardError;
    public double ConfidenceIntervalHigh  => OptionPrice + 1.96 * StandardError;
    public double Delta                   { get; set; }
    public double Gamma                   { get; set; }
    public double Vega                    { get; set; }
    public double Theta                   { get; set; }
    public double Rho                     { get; set; }
    public bool   StandardErrorAvailable  { get; set; } = true;
}
class MonteCarloEngine
{
    private readonly MarketData     _market;
    private readonly YieldCurve     _curve;
    private readonly NormalGenerator _generator;

    public MonteCarloEngine(MarketData market, YieldCurve curve,
                             NormalGenerator generator)
    {
        _market    = market    ?? throw new ArgumentNullException(nameof(market));
        _curve     = curve     ?? throw new ArgumentNullException(nameof(curve));
        _generator = generator ?? throw new ArgumentNullException(nameof(generator));
    }
    //   S_{t+dt} = S_t × exp((r − q − σ²/2)dt + σ√dt × Z)
    private double SimulateGBMPath(double T, int steps)
    {
        double r      = _curve.GetRate(T);
        double dt     = T / steps;
        double drift  = (r - _market.DividendYield
                           - 0.5 * _market.Volatility * _market.Volatility) * dt;
        double diffusion = _market.Volatility * Math.Sqrt(dt);
        double S      = _market.SpotPrice;

        for (int i = 0; i < steps; i++)
            S *= Math.Exp(drift + diffusion * _generator.GenerateStandard());

        return S;
    }

    private double SimulateGBMPathFromZ(double firstZ, double T, int steps)
    {
        double r      = _curve.GetRate(T);
        double dt     = T / steps;
        double drift  = (r - _market.DividendYield
                           - 0.5 * _market.Volatility * _market.Volatility) * dt;
        double diffusion = _market.Volatility * Math.Sqrt(dt);

        // First step uses the supplied Z (enables antithetic pairing)
        double S = _market.SpotPrice * Math.Exp(drift + diffusion * firstZ);

        // Remaining steps use fresh standard draws
        for (int i = 1; i < steps; i++)
            S *= Math.Exp(drift + diffusion * _generator.GenerateStandard());

        return S;
    }

    //   S_T = S_0 × exp((r − q − σ²/2)T + σ√T × Z)
    private double SimulateGBMSingleStep(double Z, double T)
    {
        double r         = _curve.GetRate(T);
        double drift     = (r - _market.DividendYield
                              - 0.5 * _market.Volatility * _market.Volatility) * T;
        double diffusion = _market.Volatility * Math.Sqrt(T);
        return _market.SpotPrice * Math.Exp(drift + diffusion * Z);
    }
    private (double price, double stdError) RunStandardMC(
        Option opt, int numSimulations, int numSteps)
    {
        double discountFactor = _curve.DiscountFactor(opt.TimeToExpiry);
        double sum   = 0.0;
        double sumSq = 0.0;

        for (int i = 0; i < numSimulations; i++)
        {
            double terminalPrice = SimulateGBMPath(opt.TimeToExpiry, numSteps);
            double payoff        = opt.ComputePayoff(terminalPrice);  // polymorphism
            sum   += payoff;
            sumSq += payoff * payoff;
        }

        double mean     = sum / numSimulations;
        double variance = sumSq / numSimulations - mean * mean;
        double stdError = Math.Sqrt(Math.Max(variance, 0.0) / numSimulations);

        return (discountFactor * mean, discountFactor * stdError);
    }
    private (double price, double stdError) RunAntitheticMC(
        Option opt, int numSimulations, int numSteps)
    {
        double discountFactor = _curve.DiscountFactor(opt.TimeToExpiry);
        int    numPairs       = numSimulations / 2;
        double sum   = 0.0;
        double sumSq = 0.0;

        for (int i = 0; i < numPairs; i++)
        {
            // Draw one antithetic pair: (Z, −Z)
            var (z, zAntithetic) = _generator.GenerateAntithetic();

            // Simulate two paths using the paired draws
            double priceFromZ    = SimulateGBMPathFromZ(z,           opt.TimeToExpiry, numSteps);
            double priceFromZAnt = SimulateGBMPathFromZ(zAntithetic, opt.TimeToExpiry, numSteps);

            // Average the two payoffs
            double pairAverage = 0.5 * (opt.ComputePayoff(priceFromZ)    
                                       + opt.ComputePayoff(priceFromZAnt));

            sum   += pairAverage;
            sumSq += pairAverage * pairAverage;
        }

        double mean     = sum / numPairs;
        double variance = sumSq / numPairs - mean * mean;

        double stdError = Math.Sqrt(Math.Max(variance, 0.0) / numPairs);

        return (discountFactor * mean, discountFactor * stdError);
    }
    public (double price, double stdError) ComputePrice(
        Option opt, int numSimulations, int numSteps, bool useAntithetic)
    {
        return useAntithetic
            ? RunAntitheticMC  (opt, numSimulations, numSteps)
            : RunStandardMC    (opt, numSimulations, numSteps);
    }

    public double ComputeDelta(Option opt, int numSimulations,
                                int numSteps, bool useAntithetic)
    {
        double h        = _market.SpotPrice * 0.01;
        double original = _market.SpotPrice;

        _market.SpotPrice = original + h;
        double priceUp    = ComputePrice(opt, numSimulations, numSteps, useAntithetic).price;

        _market.SpotPrice = original - h;
        double priceDown  = ComputePrice(opt, numSimulations, numSteps, useAntithetic).price;

        _market.SpotPrice = original;   

        return (priceUp - priceDown) / (2.0 * h);
    }

    public double ComputeGamma(Option opt, int numSimulations,
                                int numSteps, bool useAntithetic)
    {
        double h        = _market.SpotPrice * 0.01;
        double original = _market.SpotPrice;

        _market.SpotPrice = original + h;
        double priceUp    = ComputePrice(opt, numSimulations, numSteps, useAntithetic).price;

        _market.SpotPrice = original;
        double priceMid   = ComputePrice(opt, numSimulations, numSteps, useAntithetic).price;

        _market.SpotPrice = original - h;
        double priceDown  = ComputePrice(opt, numSimulations, numSteps, useAntithetic).price;

        _market.SpotPrice = original;   // restore

        return (priceUp - 2.0 * priceMid + priceDown) / (h * h);
    }

    public double ComputeVega(Option opt, int numSimulations,
                               int numSteps, bool useAntithetic)
    {
        double h        = 0.0001;                    // 1 basis point vol bump
        double original = _market.Volatility;

        _market.Volatility = original + h;
        double priceUp     = ComputePrice(opt, numSimulations, numSteps, useAntithetic).price;

        _market.Volatility = original - h;
        double priceDown   = ComputePrice(opt, numSimulations, numSteps, useAntithetic).price;

        _market.Volatility = original;               // restore

        return (priceUp - priceDown) / (2.0 * h) * 0.01;   // per 1% vol move
    }

    public double ComputeTheta(Option opt, int numSimulations,
                                int numSteps, bool useAntithetic)
    {
        double oneDayInYears = 1.0 / 365.0;

        double priceToday = ComputePrice(opt, numSimulations, numSteps, useAntithetic).price;
        Option shorterOption = opt is EuropeanCall
            ? (Option) new EuropeanCall(opt.StrikePrice, opt.TimeToExpiry - oneDayInYears)
            : (Option) new EuropeanPut (opt.StrikePrice, opt.TimeToExpiry - oneDayInYears);

        double priceTomorrow = ComputePrice(shorterOption, numSimulations,
                                            numSteps, useAntithetic).price;

        return priceTomorrow - priceToday;          // negative = time decay
    }

    public double ComputeRho(Option opt, int numSimulations,
                              int numSteps, bool useAntithetic)
    {
        double h = 0.0001;                           // 1 basis point rate bump

        double baseRate = _curve.GetRate(opt.TimeToExpiry);

        _curve.AddRatePillar(opt.TimeToExpiry, baseRate + h);
        double priceUp = ComputePrice(opt, numSimulations, numSteps, useAntithetic).price;

        _curve.AddRatePillar(opt.TimeToExpiry, baseRate - h);
        double priceDown = ComputePrice(opt, numSimulations, numSteps, useAntithetic).price;

        _curve.AddRatePillar(opt.TimeToExpiry, baseRate); // restore

        return (priceUp - priceDown) / (2.0 * h) * 0.0001;  // per 1bp move
    }
    public PricingResult RunVanDerCorput(Option opt, int numSimulations,
                                          int base1, int base2)
    {
        double discountFactor = _curve.DiscountFactor(opt.TimeToExpiry);
        double sum = 0.0;
        for (int i = 1; i <= numSimulations; i++)
        {
            double Z             = NormalGenerator.VanDerCorputNormal(i, base1, base2);
            double terminalPrice = SimulateGBMSingleStep(Z, opt.TimeToExpiry);
            sum += opt.ComputePayoff(terminalPrice);  
        }

        double optionPrice = discountFactor * sum / numSimulations;

        
        double delta = ComputeVdCDelta(opt, numSimulations, base1, base2);
        double gamma = ComputeVdCGamma(opt, numSimulations, base1, base2);
        double vega  = ComputeVdCVega (opt, numSimulations, base1, base2);
        double theta = ComputeVdCTheta(opt, numSimulations, base1, base2);

        return new PricingResult
        {
            OptionPrice             = optionPrice,
            StandardError           = 0.0,          
            StandardErrorAvailable  = false,
            Delta                   = delta,
            Gamma                   = gamma,
            Vega                    = vega,
            Theta                   = theta,
            Rho                     = 0.0          
        };
    }
    private double VdCPrice(Option opt, int numSimulations, int base1, int base2)
    {
        double df  = _curve.DiscountFactor(opt.TimeToExpiry);
        double sum = 0.0;
        for (int i = 1; i <= numSimulations; i++)
        {
            double Z = NormalGenerator.VanDerCorputNormal(i, base1, base2);
            sum += opt.ComputePayoff(SimulateGBMSingleStep(Z, opt.TimeToExpiry));
        }
        return df * sum / numSimulations;
    }

    private double ComputeVdCDelta(Option opt, int n, int b1, int b2)
    {
        double h = _market.SpotPrice * 0.01;
        double s = _market.SpotPrice;
        _market.SpotPrice = s + h; double up   = VdCPrice(opt, n, b1, b2);
        _market.SpotPrice = s - h; double down = VdCPrice(opt, n, b1, b2);
        _market.SpotPrice = s;
        return (up - down) / (2.0 * h);
    }

    private double ComputeVdCGamma(Option opt, int n, int b1, int b2)
    {
        double h = _market.SpotPrice * 0.01;
        double s = _market.SpotPrice;
        _market.SpotPrice = s + h; double up  = VdCPrice(opt, n, b1, b2);
        _market.SpotPrice = s;     double mid = VdCPrice(opt, n, b1, b2);
        _market.SpotPrice = s - h; double dn  = VdCPrice(opt, n, b1, b2);
        _market.SpotPrice = s;
        return (up - 2.0 * mid + dn) / (h * h);
    }

    private double ComputeVdCVega(Option opt, int n, int b1, int b2)
    {
        double h = 0.0001;
        double v = _market.Volatility;
        _market.Volatility = v + h; double up   = VdCPrice(opt, n, b1, b2);
        _market.Volatility = v - h; double down = VdCPrice(opt, n, b1, b2);
        _market.Volatility = v;
        return (up - down) / (2.0 * h) * 0.01;
    }

    private double ComputeVdCTheta(Option opt, int n, int b1, int b2)
    {
        double today    = VdCPrice(opt, n, b1, b2);
        Option shorter  = opt is EuropeanCall
            ? (Option) new EuropeanCall(opt.StrikePrice, opt.TimeToExpiry - 1.0/365.0)
            : (Option) new EuropeanPut (opt.StrikePrice, opt.TimeToExpiry - 1.0/365.0);
        double tomorrow = VdCPrice(shorter, n, b1, b2);
        return tomorrow - today;
    }
    public PricingResult RunFullPricing(Option opt, int numSimulations,
                                         int numSteps, bool useAntithetic)
    {
        Console.WriteLine("  Computing price...");
        var (price, se) = ComputePrice(opt, numSimulations, numSteps, useAntithetic);

        Console.WriteLine("  Computing Delta...");
        double delta = ComputeDelta(opt, numSimulations / 2, numSteps, useAntithetic);

        Console.WriteLine("  Computing Gamma...");
        double gamma = ComputeGamma(opt, numSimulations / 2, numSteps, useAntithetic);

        Console.WriteLine("  Computing Vega...");
        double vega  = ComputeVega (opt, numSimulations / 2, numSteps, useAntithetic);

        Console.WriteLine("  Computing Theta...");
        double theta = ComputeTheta(opt, numSimulations / 2, numSteps, useAntithetic);

        Console.WriteLine("  Computing Rho...");
        double rho   = ComputeRho  (opt, numSimulations / 2, numSteps, useAntithetic);

        return new PricingResult
        {
            OptionPrice            = price,
            StandardError          = se,
            StandardErrorAvailable = true,
            Delta                  = delta,
            Gamma                  = gamma,
            Vega                   = vega,
            Theta                  = theta,
            Rho                    = rho
        };
    }
}
static class ConsoleInterface
{
    public static double ReadDouble(string prompt, double min = double.MinValue,
                                                    double max = double.MaxValue)
    {
        while (true)
        {
            Console.Write(prompt);
            if (double.TryParse(Console.ReadLine() ?? "", out double value)
                && value >= min && value <= max)
                return value;
            Console.WriteLine($"  Please enter a number between {min} and {max}.");
        }
    }

    public static int ReadInt(string prompt, int min = 1, int max = int.MaxValue)
    {
        while (true)
        {
            Console.Write(prompt);
            if (int.TryParse(Console.ReadLine() ?? "", out int value)
                && value >= min && value <= max)
                return value;
            Console.WriteLine($"  Please enter a whole number between {min} and {max}.");
        }
    }

    public static bool ReadYesNo(string prompt)
    {
        while (true)
        {
            Console.Write(prompt + " (Y/N): ");
            string input = (Console.ReadLine() ?? "").Trim().ToUpper();
            if (input == "Y") return true;
            if (input == "N") return false;
            Console.WriteLine("  Please enter Y or N.");
        }
    }

    public static void PrintResult(PricingResult result, string label,
                                    string method)
    {
        Console.WriteLine();
        Console.WriteLine($" Method: {method,-40} ║");
        Console.WriteLine($" Price  : {result.OptionPrice,10:F4}");

        if (result.StandardErrorAvailable)
        {
            Console.WriteLine($"  Std Err: {result.StandardError,10:F4}                           ");
            Console.WriteLine($"  95% CI : [{result.ConfidenceIntervalLow:F4},  {result.ConfidenceIntervalHigh:F4}]           ║");
        }
        else
        {
            Console.WriteLine($"  Std Err: {"N/A (quasi-MC)",-10}");
        }
        Console.WriteLine(".     Greeks:                                         ");
        Console.WriteLine($"     Delta : {result.Delta,10:F4}                           ");
        Console.WriteLine($"     Gamma : {result.Gamma,10:F4}                           ");
        Console.WriteLine($"    Vega  : {result.Vega,10:F4}  (per 1% vol move)   ");
        Console.WriteLine($"     Theta : {result.Theta,10:F4}  (per calendar day)  ");
        if (result.StandardErrorAvailable)
            Console.WriteLine($"    Rho   : {result.Rho,10:F6}  (per 1bp rate move) ");
    }
}

class Program
{
    static void Main()
    {
        Console.WriteLine();
        Console.WriteLine("        Monte Carlo Option Pricing Engine          ");
        Console.WriteLine("    Antithetic Variance Reduction + Van der Corput  ");
        Console.WriteLine();

        Console.WriteLine("Underlying Parameters ");
        double spot  = ConsoleInterface.ReadDouble("  Spot price S₀              : ", min: 0.01);
        double vol   = ConsoleInterface.ReadDouble("  Volatility   : ", min: 0.001, max: 5.0);
        double divq  = ConsoleInterface.ReadDouble("  Dividend yield: ", min: 0.0,   max: 1.0);
        double rate  = ConsoleInterface.ReadDouble("  Risk-free rate r            : ");
        Console.WriteLine();
        Console.WriteLine("  Option Parameters");
        double strike = ConsoleInterface.ReadDouble("  Strike price K              : ", min: 0.01);
        double expiry = ConsoleInterface.ReadDouble("  Time to expiry T (years)    : ", min: 0.01);

        Console.Write("  Call or Put?  (C/P)         : ");
        bool isCall = (Console.ReadLine() ?? "").Trim().ToUpper() == "C";

        //  Monte Carlo parameters
        Console.WriteLine();
        Console.WriteLine(" Simulation Parameters");
        int numSimulations = ConsoleInterface.ReadInt("  Number of simulations       : ", min: 1000);
        int numSteps       = ConsoleInterface.ReadInt("  Steps per path              : ", min: 1);

        //  Method selection
        Console.WriteLine();
        Console.WriteLine(" Method Selection ");
        bool useAntithetic = ConsoleInterface.ReadYesNo("  Use antithetic variance reduction?");
        bool runVdC        = ConsoleInterface.ReadYesNo("  Run Van der Corput quasi-MC?");

        int vdcBase1 = 2, vdcBase2 = 3, vdcSimulations = numSimulations;
        if (runVdC)
        {
            Console.WriteLine("  (Choose two coprime bases)");
            vdcBase1       = ConsoleInterface.ReadInt("  VdC base 1                  : ", min: 2);
            vdcBase2       = ConsoleInterface.ReadInt("  VdC base 2                  : ", min: 2);
            vdcSimulations = ConsoleInterface.ReadInt("  VdC number of simulations   : ", min: 10);
        }

        // ── Build objects ────────────────────────────────────────────────────
        var market    = new MarketData(spot, vol, divq);
        var curve     = new YieldCurve();
        curve.AddRatePillar(expiry, rate);

        var generator = new NormalGenerator();
        var engine    = new MonteCarloEngine(market, curve, generator);

        Option option = isCall
            ? (Option) new EuropeanCall(strike, expiry)
            : (Option) new EuropeanPut (strike, expiry);

        string methodLabel = useAntithetic ? "Antithetic Monte Carlo"
                                           : "Standard Monte Carlo";

        Console.WriteLine();
        Console.WriteLine($"  Running {methodLabel}...");

        PricingResult mcResult = engine.RunFullPricing(
            option, numSimulations, numSteps, useAntithetic);

        ConsoleInterface.PrintResult(mcResult, option.Label, methodLabel);

        if (runVdC)
        {
            Console.WriteLine();
            Console.WriteLine($"  Running Van der Corput (base {vdcBase1}, {vdcBase2})...");

            PricingResult vdcResult = engine.RunVanDerCorput(
                option, vdcSimulations, vdcBase1, vdcBase2);

            ConsoleInterface.PrintResult(
                vdcResult, option.Label,
                $"Van der Corput  base₁={vdcBase1}  base₂={vdcBase2}");
        }
        Console.WriteLine();
        Console.WriteLine("  Press any key to exit...");
        Console.ReadKey();
    }
}
