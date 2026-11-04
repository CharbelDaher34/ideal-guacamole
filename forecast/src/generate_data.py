# Sample data with the columns and links of the 17 tables in the data requirements.
# Every client gets a monthly "activity level" (size x growth x season x noise); each category's
# money in / out is drawn from it, then split into individual postings on real days.
# Payments that are scheduled in advance (loans, standing orders, deposits, letters of credit,
# card statements, SADAD bills, post-dated cheques) are created as schedule rows first and their
# postings carry origin_type + origin_ref, so the flow can tell which money was already known.
import hashlib
import hmac

import numpy as np
import pandas as pd

HMAC_KEY = b"sample-key-kept-by-warehouse-team"   # stands in for the secret key (never shared with analysts)

SEGMENTS = {  # segment: (share of clients, median monthly money in, SAR)
    "SME": (0.60, 400_000), "MID_CORPORATE": (0.30, 3_000_000), "LARGE_CORPORATE": (0.10, 18_000_000)}
SECTORS = {  # sector: (industry, takes card payments, imports with letters of credit)
    "RETAIL": ("Grocery and general retail", True, False),
    "HOSPITALITY": ("Restaurants and hotels", True, False),
    "HEALTHCARE": ("Clinics and pharmacies", True, False),
    "WHOLESALE": ("Wholesale trade", False, True),
    "MANUFACTURING": ("Food and building materials", False, True),
    "CONSTRUCTION": ("Building contractors", False, False),
    "TRANSPORT": ("Transport and logistics", False, False),
    "IT_SERVICES": ("IT and business services", False, False),
    "REAL_ESTATE": ("Property management", False, False),
}
CITIES = [("RIYADH", "Riyadh"), ("MAKKAH", "Jeddah"), ("EASTERN", "Dammam"), ("EASTERN", "Khobar"),
          ("MADINAH", "Madinah"), ("QASSIM", "Buraidah"), ("ASIR", "Abha")]
RAMADAN = {(2021, 4), (2022, 4), (2023, 3), (2024, 3), (2025, 3), (2026, 2), (2027, 2)}
SEASON = np.array([1.00, 0.98, 1.03, 1.02, 1.00, 0.97, 0.88, 0.90, 1.00, 1.03, 1.05, 1.14])  # Jan..Dec
FINANCING_PRODUCTS = ["MURABAHA", "TAWARRUQ", "IJARA", "BEI_AL_AJAL", "DIMINISHING_MUSHARAKA", "LC_MURABAHA",
                      "PROJECT_FINANCE"]
BILLERS = {  # biller id: (name, category, share of the monthly level, frequency in months)
    "B001": ("Saudi Electricity Company", "ELECTRICITY", 0.012, 1),
    "B002": ("National Water Company", "WATER", 0.002, 1),
    "B003": ("STC Business", "TELECOM", 0.002, 1),
    "B004": ("GOSI", "GOVERNMENT", None, 1),          # social insurance: a share of payroll
    "B005": ("ZATCA VAT", "GOVERNMENT", 0.035, 3),
    "B006": ("ZATCA Zakat", "GOVERNMENT", 0.05, 12),
    "B007": ("Balady municipality", "GOVERNMENT", 0.004, 12),
}


def code_iban(iban):
    """Counterparty IBAN replaced by an HMAC-SHA256 code: the same IBAN always gets the same code."""
    return hmac.new(HMAC_KEY, iban.encode(), hashlib.sha256).hexdigest()[:24]


def fake_iban(rng):
    return code_iban("SA" + "".join(map(str, rng.integers(0, 10, 22))))


class Postings:
    """Collects account postings as column arrays (fast for hundreds of thousands of rows)."""
    COLUMNS = ["account_no", "value_date", "amount_orig", "currency", "exchange_rate", "debit_credit",
               "txn_code", "channel", "txn_type_tag", "counterparty_iban_code", "counterparty_bank",
               "counterparty_name", "own_account_flag", "origin_type", "origin_ref"]

    def __init__(self):
        self.parts = []

    def add(self, account, dates, amounts, dc, code, channel, tag, origin_type="", origin_ref="",
            cp_iban="", cp_bank="", cp_name="", own=False, currency="SAR", rate=1.0):
        dates = np.atleast_1d(np.asarray(dates, dtype="datetime64[D]"))
        n = len(dates)
        if n == 0:
            return
        full = lambda v: np.full(n, v, dtype=object) if np.ndim(v) == 0 else np.asarray(v, dtype=object)
        self.parts.append([full(account), dates, np.round(np.atleast_1d(amounts).astype(float), 2), full(currency),
                           np.full(n, rate), full(dc), full(code), full(channel), full(tag), full(cp_iban),
                           full(cp_bank), full(cp_name), np.full(n, own), full(origin_type), full(origin_ref)])

    def frame(self):
        cols = [np.concatenate([p[i] for p in self.parts]) for i in range(len(self.COLUMNS))]
        return pd.DataFrame(dict(zip(self.COLUMNS, cols)))


def month_days(month, rng, n, first=1, last=None):
    """n random days in a month (a pandas Period)."""
    last = last or month.days_in_month
    return (month.start_time + pd.to_timedelta(rng.integers(first, last + 1, n) - 1, unit="D")).values


def split(total, n, rng):
    """Split an amount into n positive parts."""
    if n <= 1:
        return np.array([total])
    return total * rng.dirichlet(np.full(n, 1.5))


def generate(n_clients=300, last_month="2026-10", seed=7, max_history_months=60):
    rng = np.random.default_rng(seed)
    last = pd.Period(last_month, "M")
    today = (last + 1).start_time                       # as-of date: first day after the last full month
    data_start = last - max_history_months + 1
    bank_codes = ["RJHI", "NCBK", "RIBL", "SABB", "BSFR", "ARNB", "ALBI"]

    clients, accounts, payroll, loans, deposits, orders = [], [], [], [], [], []
    sadad, trade, card_sales, corp_cards, invoices, cheques, groups = [], [], [], [], [], [], []
    post = Postings()
    counters = {"acc": 0, "file": 0, "loan": 0, "dep": 0, "so": 0, "bill": 0, "lc": 0, "stmt": 0, "inv": 0, "chq": 0}

    def next_id(kind, prefix, width=7):
        counters[kind] += 1
        return f"{prefix}{counters[kind]:0{width}d}"

    def new_account(client_no, product, code, opened, restricted=False, payroll_flag=False):
        acc = f"0108{next_id('acc', '', 8)}"
        accounts.append({"account_no": acc, "client_no": client_no, "product_type": product, "product_code": code,
                         "currency": "SAR", "open_date": opened, "close_date": pd.NaT, "status": "ACTIVE",
                         "restricted_flag": restricted, "payroll_account_flag": payroll_flag})
        return acc

    segments = list(SEGMENTS)
    seg_p = [SEGMENTS[s][0] for s in segments]
    for i in range(n_clients):
        client_no = f"{3100000 + i * 7 + 3}"
        segment = rng.choice(segments, p=seg_p)
        sector = rng.choice(list(SECTORS))
        industry, takes_cards, imports = SECTORS[sector]
        region, city = CITIES[rng.integers(len(CITIES))]
        level0 = SEGMENTS[segment][1] * rng.lognormal(0, 0.45)
        history = int(rng.choice([rng.integers(6, 12), rng.integers(12, 24), rng.integers(24, 36),
                                  rng.integers(36, 48), rng.integers(48, max_history_months + 1)],
                                 p=[0.08, 0.25, 0.25, 0.22, 0.20]))
        first = last - history + 1
        joined = first.start_time + pd.Timedelta(days=int(rng.integers(0, 20)))
        if history >= max_history_months:
            joined -= pd.Timedelta(days=int(rng.integers(0, 3650)))
            first = max(first, data_start)
        closed = rng.random() < 0.05 and history > 18
        end = last - int(rng.integers(1, 9)) if closed else last
        employees = max(3, int(level0 / rng.uniform(25_000, 60_000)))
        salary = rng.uniform(6_000, 14_000)
        clients.append({
            "client_no": client_no, "party_type": "Business Customer", "segment": segment,
            "sub_segment": f"{segment}_{'A' if level0 > SEGMENTS[segment][1] else 'B'}", "sector": sector,
            "industry": industry, "region": region, "city": city, "joining_date": joined,
            "legal_form": rng.choice(["LLC", "JSC", "SOLE_ESTABLISHMENT", "PARTNERSHIP"], p=[0.6, 0.1, 0.25, 0.05]),
            "sole_proprietor_flag": False, "last_annual_sales": round(level0 * 12 * rng.uniform(0.9, 1.1), -3),
            "number_of_employees": employees, "parent_client_no": None,
            "status": "CLOSED" if closed else "ACTIVE",
            "closure_date": (end + 1).start_time + pd.Timedelta(days=int(rng.integers(0, 25))) if closed else pd.NaT})
        if i > 10 and rng.random() < 0.08:
            parent = clients[int(rng.integers(0, i))]["client_no"]
            clients[-1]["parent_client_no"] = parent
            groups.append({"parent_client_no": parent, "child_client_no": client_no,
                           "relationship_type": "SUBSIDIARY", "start_date": joined})

        opened = joined
        main = new_account(client_no, "CURRENT", "CA01", opened)
        pay_acc = new_account(client_no, "CURRENT", "CA02", opened, payroll_flag=True) if employees > 40 else main
        if pay_acc == main:
            accounts[-1]["payroll_account_flag"] = True
        savings = new_account(client_no, "CALL_INVESTMENT", "CI01", opened) if rng.random() < 0.3 else None
        escrow = new_account(client_no, "CASH_MARGIN", "CM01", opened, restricted=True) if imports else None
        if closed:
            for a in accounts:
                if a["client_no"] == client_no:
                    a.update(status="CLOSED", close_date=clients[-1]["closure_date"])

        months = pd.period_range(first, end, freq="M")
        n = len(months)
        growth = rng.normal(0.006, 0.008)
        noise = rng.uniform(0.08, 0.30)
        season = np.array([SEASON[m.month - 1] for m in months])
        ramadan = np.array([1.0 + (0.25 if takes_cards else 0.08) * ((m.year, m.month) in RAMADAN) for m in months])
        level = level0 * (1 + growth) ** np.arange(n) * season * ramadan * rng.lognormal(0, noise, n)
        txn_scale = {"SME": 1, "MID_CORPORATE": 2, "LARGE_CORPORATE": 3}[segment]
        customers = [fake_iban(rng) for _ in range(20 * txn_scale)]
        suppliers = [fake_iban(rng) for _ in range(8 * txn_scale)]
        supplier_names = [f"Supplier {rng.integers(1000, 9999)} Trading Co." for _ in suppliers]

        # ---- schedules decided up front --------------------------------------------------------
        # loans: 0-2 per client, some started before the data window, schedules run into the future
        for _ in range(rng.choice([0, 1, 2], p=[0.45, 0.4, 0.15])):
            loan_no = next_id("loan", "LN", 8)
            start = (first - int(rng.integers(-n + 1, 24))).start_time + pd.Timedelta(days=int(rng.integers(0, 27)))
            tenor = int(rng.choice([12, 24, 36, 48, 60]))
            every = 1 if rng.random() < 0.7 else 3
            principal = round(level0 * rng.uniform(1.5, 10), -3)
            rate = rng.uniform(0.05, 0.08)
            limit = round(principal * rng.uniform(1.0, 1.4), -3)
            product = rng.choice(FINANCING_PRODUCTS, p=[0.3, 0.25, 0.1, 0.1, 0.1, 0.1, 0.05])
            settle_early = rng.random() < 0.05
            due_dates = [start + pd.DateOffset(months=k) for k in range(every, tenor + 1, every)]
            outstanding = principal
            settled = False
            for k, due in enumerate(due_dates, 1):
                if settled:
                    break
                pr = principal / len(due_dates)
                profit = outstanding * rate * every / 12
                status, paid_date, paid = "NOT_DUE", pd.NaT, np.nan
                early = settle_early and due < today and k == len(due_dates) // 2
                if early:
                    pr, settled = outstanding, True
                if due < today:
                    late = rng.random() < 0.07
                    paid_date = due + pd.Timedelta(days=int(rng.integers(3, 40)) if late else 0)
                    status = "PAID_LATE" if late else "PAID"
                    if paid_date >= today:
                        paid_date, status = pd.NaT, "OVERDUE"
                    paid = pr + profit if pd.notna(paid_date) else np.nan
                loans.append({"loan_no": loan_no, "client_no": client_no, "product": product, "account_debited": main,
                              "contract_start_date": start, "maturity_date": due_dates[-1], "instalment_no": k,
                              "due_date": due, "principal": round(pr, 2), "profit": round(profit, 2),
                              "instalment_amount": round(pr + profit, 2), "paid_date": paid_date,
                              "paid_amount": round(paid, 2) if pd.notna(paid) else np.nan, "status": status,
                              "early_settlement_flag": bool(early), "credit_limit_total": limit,
                              "credit_limit_used": round(outstanding, 2),
                              "credit_limit_available": round(limit - outstanding, 2)})
                outstanding -= pr
                if pd.notna(paid_date) and first.start_time <= paid_date <= end.end_time:
                    post.add(main, paid_date, pr + profit, "D", "LOAN_INSTL", "SYSTEM", "FINANCING", "LOAN", loan_no)

        # standing orders: rent for some renters, fixed supplier contracts, insurance
        renter = rng.random() < 0.7
        rent_year = level0 * 12 * rng.uniform(0.03, 0.08)
        rent_every = int(rng.choice([1, 3, 6, 12], p=[0.1, 0.15, 0.35, 0.4]))
        rent_offset = int(rng.integers(0, rent_every))
        rent_by_order = renter and rng.random() < 0.35
        landlord = fake_iban(rng)
        plans = []
        if rent_by_order:
            plans.append(("RENT", rent_year * rent_every / 12, rent_every, landlord, "RENT"))
        if rng.random() < 0.2:
            plans.append(("SUPPLIER", level0 * rng.uniform(0.03, 0.08), 1, suppliers[0], "SUPPLIER"))
        if rng.random() < 0.15:
            plans.append(("INSURANCE", level0 * rng.uniform(0.005, 0.015), 1, code_iban("SA-insurer"), "OTHER"))
        for beneficiary, amount, every, iban, tag in plans:
            so = next_id("so", "SO", 7)
            created = first.start_time + pd.Timedelta(days=int(rng.integers(0, 20)))
            start = (first + (rent_offset if beneficiary == "RENT" else 0)).start_time + pd.Timedelta(days=int(rng.integers(0, 5)))
            stop = end.end_time if closed else pd.NaT
            pay_days = pd.date_range(start, (end + 6).end_time if not closed else stop, freq=pd.DateOffset(months=every))
            nxt = [d for d in pay_days if d >= today]
            orders.append({"instruction_id": so, "account_no": main, "beneficiary_type": beneficiary,
                           "beneficiary_iban_code": iban, "amount": round(amount, 2), "currency": "SAR",
                           "frequency": {1: "MONTHLY", 3: "QUARTERLY", 6: "SEMI_ANNUAL", 12: "ANNUAL"}[every],
                           "start_date": start, "end_date": stop, "next_payment_date": nxt[0] if nxt else pd.NaT,
                           "status": "CANCELLED" if closed else "ACTIVE", "created_date": created,
                           "changed_date": created})
            past = pay_days[pay_days < min(today, end.end_time)]
            post.add(main, past.values, np.full(len(past), amount), "D", "SO_PMT", "SYSTEM", tag, "STANDING_ORDER", so,
                     iban, rng.choice(bank_codes))

        # time deposits placed every few months
        if savings is not None or rng.random() < 0.15:
            d = first.start_time + pd.Timedelta(days=int(rng.integers(10, 90)))
            while d < min(today, end.end_time):
                tenor = int(rng.choice([1, 3, 6]))
                amount = round(level0 * rng.uniform(0.5, 2.5), -3)
                maturity = d + pd.DateOffset(months=tenor)
                profit = round(amount * rng.uniform(0.04, 0.055) * tenor / 12, 2)
                dep = next_id("dep", "TD", 7)
                rollover = rng.random() < 0.25
                deposits.append({"deposit_id": dep, "client_no": client_no, "account_no": main,
                                 "product": rng.choice(["TIME_DEPOSIT", "MUDARABA", "CALL_INVESTMENT"], p=[0.6, 0.3, 0.1]),
                                 "amount": amount, "start_date": d, "maturity_date": maturity, "expected_profit": profit,
                                 "rollover_instruction": "ROLLOVER" if rollover else "CREDIT_ACCOUNT",
                                 "status": "MATURED" if maturity < today else "ACTIVE"})
                post.add(main, d, amount, "D", "DEP_PLACE", "ONLINE", "OWN", "DEPOSIT", dep, own=True)
                if not rollover and maturity < min(today, end.end_time):
                    post.add(main, maturity, amount + profit, "C", "DEP_MAT", "SYSTEM", "OWN", "DEPOSIT", dep)
                d = maturity + pd.Timedelta(days=int(rng.integers(20, 120)))

        # ---- month by month --------------------------------------------------------------------
        pay_level = employees * salary
        for t, m in enumerate(months):
            L = level[t]
            dim = m.days_in_month
            posted_until = min(end.end_time, today - pd.Timedelta(days=1))

            # money in
            card_share = rng.uniform(0.25, 0.45) if takes_cards else 0.0
            receipts = L * (1 - card_share) * rng.uniform(0.85, 0.97)
            k = int(rng.integers(5, 15) * txn_scale)
            amounts = split(receipts, k, rng)
            cps = rng.choice(len(customers), k)
            invoice_paid = rng.random(k) < 0.3
            for j in range(k):
                day = month_days(m, rng, 1)[0]
                if invoice_paid[j]:
                    inv = next_id("inv", "INV", 8)
                    issued = pd.Timestamp(day) - pd.Timedelta(days=int(rng.integers(20, 70)))
                    invoices.append({"invoice_id": inv, "client_no": client_no, "customer_iban_code": customers[cps[j]],
                                     "issue_date": issued, "due_date": issued + pd.Timedelta(days=30),
                                     "amount": round(amounts[j], 2), "paid_date": pd.Timestamp(day),
                                     "paid_amount": round(amounts[j], 2)})
                    post.add(main, day, amounts[j], "C", "TRF_IN", "H2H", "COLLECTION", "INVOICE", inv, customers[cps[j]],
                             rng.choice(bank_codes), f"Customer {cps[j]:03d} Co.")
                else:
                    post.add(main, day, amounts[j], "C", "TRF_IN", rng.choice(["ONLINE", "H2H", "BRANCH"]),
                             "COLLECTION", cp_iban=customers[cps[j]], cp_bank=rng.choice(bank_codes),
                             cp_name=f"Customer {cps[j]:03d} Co.")
            if takes_cards:
                days = pd.date_range(m.start_time, m.end_time, freq="D")
                daily = L * card_share * rng.dirichlet(np.full(len(days), 8.0))
                fees = daily * rng.uniform(0.012, 0.02)
                charge = np.where(rng.random(len(days)) < 0.02, daily * 0.05, 0.0)
                settle = days + pd.to_timedelta(np.where(days.dayofweek == 3, 3, np.where(days.dayofweek == 4, 2, 1)), unit="D")
                net = daily - fees - charge
                merchant = f"M{client_no}"
                for d, s, a, f, c, nt in zip(days, settle, daily, fees, charge, net):
                    card_sales.append({"merchant_id": merchant, "client_no": client_no, "sales_date": d,
                                       "settlement_date": s, "sales_amount": round(a, 2), "fees": round(f, 2),
                                       "chargebacks": round(c, 2), "net_amount_paid": round(nt, 2)})
                ok = settle <= posted_until
                post.add(main, settle[ok].values, net[ok], "C", "POS_SETTLE", "POS", "CARD_SETTLEMENT", "CARD_SETTLEMENT",
                         merchant)
            if rng.random() < 0.3:
                for _ in range(int(rng.integers(1, 4))):
                    chq = next_id("chq", "CHQ", 8)
                    amount = L * rng.uniform(0.01, 0.04)
                    received = pd.Timestamp(month_days(m, rng, 1)[0])
                    post_dated = rng.random() < 0.25
                    cheque_date = received + pd.Timedelta(days=int(rng.integers(20, 60))) if post_dated else received
                    returned = rng.random() < 0.03
                    cleared = cheque_date + pd.Timedelta(days=1)
                    status = "RETURNED" if returned else ("HELD_POST_DATED" if cleared >= today else "CLEARED")
                    cheques.append({"cheque_id": chq, "account_no": main, "direction": "DEPOSITED", "status": status,
                                    "received_date": received, "cheque_date": cheque_date,
                                    "cleared_date": cleared if status == "CLEARED" else pd.NaT,
                                    "amount": round(amount, 2), "post_dated_flag": post_dated})
                    if status == "CLEARED" and cleared <= end.end_time:
                        post.add(main, cleared, amount, "C", "CHQ_DEP", "BRANCH", "CHEQUE", "CHEQUE", chq)
            if rng.random() < 0.35:
                post.add(main, month_days(m, rng, 1), L * rng.uniform(0.005, 0.03), "C", "MISC_CR", "ONLINE", "OTHER")

            # money out: payroll
            pay = pay_level * (1 + growth) ** t * rng.uniform(0.97, 1.03)
            pay_day = m.start_time + pd.Timedelta(days=int(rng.integers(23, 27)))
            files = [("SALARY", pay)]
            if (m.year, m.month) in RAMADAN and rng.random() < 0.5:
                files.append(("BONUS", pay * rng.uniform(0.3, 1.0)))
            for kind, amount in files:
                if pay_day >= today:
                    continue
                # about a quarter of files are cancelled and sent again; a few test files and confirmed duplicates
                statuses = [st for st, p in (("CANCELED", 0.25), ("TEST", 0.02)) if rng.random() < p] + ["PAID"]
                statuses += ["CNFRM_DPLCT"] if rng.random() < 0.02 else []
                for status in statuses:
                    fid = next_id("file", "PF", 9)
                    payroll.append({"FILE_ID": fid, "DBT_CSTMR_CIF": client_no, "PAYROLL_ACNT_NMBR": pay_acc,
                                    "CRT_DTTS": pay_day - pd.Timedelta(hours=int(rng.integers(1, 30))),
                                    "SLRY_TRNS_DT": pay_day, "FND_DT": pay_day, "TOT_AMT": round(amount, 2),
                                    "NMBR_OF_EMP": employees, "FILE_TYP_EN_NM": kind,
                                    "FILE_SUB_TYP_EN_NM": "WPS" if kind == "SALARY" else "NON_WPS",
                                    "STA_CD": status, "DLT_FLG": "N"})
                    if status == "PAID":
                        post.add(pay_acc, pay_day, amount, "D", "PAYROLL_DR", "H2H", "SALARY", "PAYROLL", fid)
            if pay_acc != main:
                post.add(main, pay_day - pd.Timedelta(days=1), pay * 1.02, "D", "OWN_TRF", "ONLINE", "OWN", own=True)
                post.add(pay_acc, pay_day - pd.Timedelta(days=1), pay * 1.02, "C", "OWN_TRF", "ONLINE", "OWN", own=True)

            # rent paid by hand (renters without a standing order)
            if renter and not rent_by_order and (t - rent_offset) % rent_every == 0:
                post.add(main, month_days(m, rng, 1, 1, 10), rent_year * rent_every / 12 * rng.uniform(0.98, 1.02),
                         "D", "TRF_OUT", "ONLINE", "RENT", cp_iban=landlord, cp_bank=rng.choice(bank_codes))

            # SADAD bills: issued about two weeks before the due date, a few paid late or still open
            for biller, (bname, bcat, share, every) in BILLERS.items():
                if (m.month - 1) % every != (0 if every < 12 else 3) % every:
                    continue
                amount = pay * 0.1175 if share is None else L * share * (1.8 if bcat == "ELECTRICITY" and m.month in (6, 7, 8, 9) else 1.0)
                amount *= rng.uniform(0.9, 1.1)
                due = m.start_time + pd.Timedelta(days=int(rng.integers(14, dim)))
                issued = due - pd.Timedelta(days=int(rng.integers(10, 25)))
                if issued >= today:
                    continue
                paid = due + pd.Timedelta(days=int(rng.integers(3, 20)) if rng.random() < 0.1 else -int(rng.integers(0, 3)))
                open_bill = paid >= today
                sadad.append({"payment_id": None if open_bill else f"SP{counters['bill'] + 1:08d}",
                              "account_no": main, "client_no": client_no, "biller_id": biller, "biller_name": bname,
                              "biller_category": bcat, "bill_number": next_id("bill", "BL", 9),
                              "bill_issue_date": issued, "amount": round(amount, 2), "due_date": due,
                              "paid_date": pd.NaT if open_bill else paid,
                              "channel": None if open_bill else rng.choice(["ONLINE", "H2H"])})
                if not open_bill and paid <= end.end_time:
                    post.add(main, paid, amount, "D", "SADAD_PMT", sadad[-1]["channel"], "UTILITY" if bcat != "GOVERNMENT" else "GOV",
                             "SADAD", sadad[-1]["bill_number"])

            # corporate card: statement on the 20th, paid on the 10th of the next month
            if t == 0:
                has_card = rng.random() < 0.5
                card_share_out = rng.uniform(0.005, 0.02)
            if has_card:
                sdate = m.start_time + pd.Timedelta(days=19)
                if sdate < today:
                    stmt = next_id("stmt", "CS", 8)
                    amount = L * card_share_out * rng.lognormal(0, 0.3)
                    due = (m + 1).start_time + pd.Timedelta(days=9)
                    paid = due if due < today else pd.NaT
                    corp_cards.append({"record_id": stmt, "record_type": "STATEMENT", "client_no": client_no,
                                       "statement_date": sdate, "amount_due": round(amount, 2), "due_date": due,
                                       "paid_date": paid, "paid_amount": round(amount, 2) if pd.notna(paid) else np.nan,
                                       "topup_date": pd.NaT, "topup_amount": np.nan})
                    if pd.notna(paid) and paid <= end.end_time:
                        post.add(main, paid, amount, "D", "CARD_PMT", "SYSTEM", "CARD", "CORP_CARD", stmt)
                if rng.random() < 0.2:
                    tid = next_id("stmt", "CT", 8)
                    tday = pd.Timestamp(month_days(m, rng, 1)[0])
                    if tday < today:
                        amount = L * rng.uniform(0.002, 0.006)
                        corp_cards.append({"record_id": tid, "record_type": "TOPUP", "client_no": client_no,
                                           "statement_date": pd.NaT, "amount_due": np.nan, "due_date": pd.NaT,
                                           "paid_date": pd.NaT, "paid_amount": np.nan, "topup_date": tday,
                                           "topup_amount": round(amount, 2)})
                        post.add(main, tday, amount, "D", "CARD_TOPUP", "ONLINE", "CARD", "CORP_CARD", tid)

            # letters of credit and guarantees for importers
            if imports and rng.random() < 0.35:
                lc = next_id("lc", "LC", 7)
                subtype = rng.choice(["SIGHT", "USANCE", "DEFERRED_PAYMENT"], p=[0.4, 0.3, 0.3])
                issued = pd.Timestamp(month_days(m, rng, 1)[0])
                pay_date = issued + pd.Timedelta(days=int(rng.integers(10, 30) if subtype == "SIGHT" else rng.integers(60, 180)))
                usd = round(level0 * rng.uniform(0.2, 0.8) / 3.75, -2)
                trade.append({"instrument_id": lc, "client_no": client_no, "instrument_type": "LC", "subtype": subtype,
                              "amount": usd, "currency": "USD", "issue_date": issued,
                              "expiry_date": pay_date + pd.Timedelta(days=30), "payment_date": pay_date,
                              "cash_margin_account": escrow, "cash_margin_amount": round(usd * 3.75 * 0.1, 2),
                              "claim_amount": np.nan, "claim_date": pd.NaT,
                              "status": "PAID" if pay_date < today else "OUTSTANDING"})
                if pay_date < today and pay_date <= end.end_time:
                    post.add(main, pay_date, usd, "D", "LC_PMT", "SYSTEM", "SUPPLIER", "LC", lc, suppliers[1],
                             "FOREIGN", supplier_names[1], currency="USD", rate=3.75)
            if rng.random() < 0.03:
                trade.append({"instrument_id": next_id("lc", "LG", 7), "client_no": client_no, "instrument_type": "LG",
                              "subtype": rng.choice(["BID", "PERFORMANCE", "ADVANCE_PAYMENT", "CUSTOMS"]),
                              "amount": round(level0 * rng.uniform(0.1, 0.5), -3), "currency": "SAR",
                              "issue_date": m.start_time, "expiry_date": m.start_time + pd.DateOffset(months=12),
                              "payment_date": pd.NaT, "cash_margin_account": escrow or main,
                              "cash_margin_amount": 0.0, "claim_amount": np.nan, "claim_date": pd.NaT,
                              "status": "ISSUED"})

            # suppliers fill the gap between money in and the other money out (big obligations squeeze them)
            fixed = pay * 1.12 + L * 0.03
            suppliers_total = max(L * 0.08, L * rng.uniform(0.76, 0.86) - fixed)
            k = int(rng.integers(4, 12) * txn_scale)
            amounts = split(suppliers_total, k, rng)
            idx = rng.choice(len(suppliers), k)
            post.add(main, month_days(m, rng, k), amounts, "D", "TRF_OUT", rng.choice(["ONLINE", "H2H"], k), "SUPPLIER",
                     cp_iban=[suppliers[j] for j in idx], cp_bank=rng.choice(bank_codes, k),
                     cp_name=[supplier_names[j] for j in idx])
            if rng.random() < 0.5:
                post.add(main, month_days(m, rng, 1), L * rng.uniform(0.005, 0.03), "D", "MISC_DR", "BRANCH", "OTHER")
            post.add(main, m.start_time + pd.Timedelta(days=dim - 1), rng.uniform(50, 400), "D", "FEE", "SYSTEM", "FEE")
            if savings is not None and rng.random() < 0.2:
                amount = L * rng.uniform(0.05, 0.2)
                post.add(main, month_days(m, rng, 1), amount, "D", "OWN_TRF", "ONLINE", "OWN", own=True)

    # ---- assemble tables -----------------------------------------------------------------------
    tx = post.frame()
    tx["value_date"] = pd.to_datetime(tx["value_date"])
    tx = tx[tx["value_date"] < today].sort_values(["value_date", "account_no"]).reset_index(drop=True)
    tx.insert(0, "txn_id", [f"T{i:010d}" for i in range(1, len(tx) + 1)])
    tx.insert(2, "posting_date", tx["value_date"])
    tx.insert(4, "amount_sar", (tx["amount_orig"] * tx["exchange_rate"]).round(2))
    tx["reversal_flag"] = False
    tx["original_ref"] = None
    rev = tx[(tx["txn_code"].isin(["TRF_IN", "TRF_OUT"])) & (rng.random(len(tx)) < 0.003)].copy()
    rev["txn_id"] = [f"R{i:010d}" for i in range(1, len(rev) + 1)]
    rev["original_ref"] = tx.loc[rev.index, "txn_id"].values
    rev["debit_credit"] = rev["debit_credit"].map({"C": "D", "D": "C"})
    rev["reversal_flag"] = True
    rev["value_date"] = rev["posting_date"] = rev["value_date"] + pd.Timedelta(days=1)
    tx = pd.concat([tx, rev[rev["value_date"] < today]], ignore_index=True)

    acc = pd.DataFrame(accounts)
    clients_df = pd.DataFrame(clients)
    balances = daily_balances(tx, acc, clients_df, today, rng)
    tables = {
        "clients": clients_df, "accounts": acc, "transactions": tx, "daily_balances": balances,
        "standing_orders": pd.DataFrame(orders), "payroll_files": pd.DataFrame(payroll),
        "financing_schedules": pd.DataFrame(loans), "deposits": pd.DataFrame(deposits),
        "sadad_payments": pd.DataFrame(sadad), "code_lists": code_lists(), "trade_finance": pd.DataFrame(trade),
        "card_sales": pd.DataFrame(card_sales), "corporate_cards": pd.DataFrame(corp_cards),
        "bills_issued": pd.DataFrame(invoices),
        "group_structure": pd.DataFrame(groups, columns=["parent_client_no", "child_client_no", "relationship_type",
                                                         "start_date"]),
        "cheques": pd.DataFrame(cheques)}
    tables["client_monthly_summary"] = monthly_summary(tx, acc, balances, tables["payroll_files"])
    return tables


def daily_balances(tx, acc, clients, today, rng):
    """One row per current account per day, from its opening day (or the data start) to yesterday."""
    current = acc[acc["product_type"] == "CURRENT"]
    signed = tx.assign(net=np.where(tx["debit_credit"] == "C", tx["amount_sar"], -tx["amount_sar"]))
    daily = signed.groupby(["account_no", "value_date"])["net"].sum()
    rows = []
    for a in current.itertuples():
        if a.account_no not in daily.index.get_level_values(0):
            continue
        flows = daily.loc[a.account_no]
        start = flows.index.min()
        stop = min(today - pd.Timedelta(days=1), a.close_date) if pd.notna(a.close_date) else today - pd.Timedelta(days=1)
        days = pd.date_range(start, stop, freq="D")
        net = flows.reindex(days, fill_value=0.0).values
        opening = abs(net[net < 0][:40].sum()) * rng.uniform(0.8, 1.5)
        ledger = opening + np.cumsum(net)
        hold = np.where(rng.random(len(days)) < 0.05, np.abs(ledger) * 0.02, 0.0)
        limit = round(opening * 0.3, -3) if rng.random() < 0.3 else 0.0
        rows.append(pd.DataFrame({"account_no": a.account_no, "balance_date": days, "ledger_balance": ledger.round(2),
                                  "available_balance": (ledger - hold + limit).round(2), "blocked_amount": 0.0,
                                  "hold_amount": hold.round(2), "overdraft_limit": limit}))
    return pd.concat(rows, ignore_index=True)


def monthly_summary(tx, acc, balances, payroll):
    """Client per month: a small slice of the ~700-column summary (no names, RM, risk or profitability)."""
    t = tx[~tx["own_account_flag"]].merge(acc[["account_no", "client_no"]], on="account_no")
    t["month"] = t["value_date"].dt.to_period("M").dt.start_time
    credit = t["debit_credit"] == "C"
    t = t.assign(txn_count_in=credit.astype(int), txn_amount_in=t["amount_sar"].where(credit, 0.0),
                 txn_count_out=(~credit).astype(int), txn_amount_out=t["amount_sar"].where(~credit, 0.0),
                 online_amount=t["amount_sar"].where(t["channel"] == "ONLINE", 0.0),
                 cheque_count=(t["txn_code"] == "CHQ_DEP").astype(int))
    out = t.groupby(["client_no", "month"])[["txn_count_in", "txn_amount_in", "txn_count_out", "txn_amount_out",
                                             "online_amount", "cheque_count"]].sum().reset_index()
    b = balances.merge(acc[["account_no", "client_no"]], on="account_no")
    b["month"] = b["balance_date"].dt.to_period("M").dt.start_time
    avg = b.groupby(["client_no", "account_no", "month"])["ledger_balance"].mean().groupby(["client_no", "month"]).sum()
    out = out.merge(avg.rename("avg_balance").reset_index(), on=["client_no", "month"], how="left")
    p = payroll[payroll["STA_CD"] == "PAID"].assign(month=lambda x: x["SLRY_TRNS_DT"].dt.to_period("M").dt.start_time)
    pay = p.pivot_table(index=["DBT_CSTMR_CIF", "month"], columns="FILE_TYP_EN_NM", values="TOT_AMT", aggfunc="sum",
                        fill_value=0).add_prefix("payroll_").reset_index().rename(columns={"DBT_CSTMR_CIF": "client_no"})
    out = out.merge(pay, on=["client_no", "month"], how="left").rename(columns=str.lower)
    for c in ("payroll_salary", "payroll_bonus"):
        out[c] = out[c].fillna(0) if c in out else 0.0
    return out.round(2)


def code_lists():
    rows = []
    def add(kind, items):
        rows.extend({"code_type": kind, "code": c, "meaning": m} for c, m in items.items())
    add("TXN_CODE", {"TRF_IN": "Incoming transfer", "TRF_OUT": "Outgoing transfer", "SO_PMT": "Standing order payment",
                     "PAYROLL_DR": "Payroll file debit", "LOAN_INSTL": "Financing instalment", "SADAD_PMT": "SADAD bill payment",
                     "POS_SETTLE": "Card sales settlement", "CHQ_DEP": "Cheque deposit", "CARD_PMT": "Corporate card payment",
                     "CARD_TOPUP": "Corporate card top-up", "DEP_PLACE": "Deposit placement", "DEP_MAT": "Deposit maturity",
                     "LC_PMT": "Letter of credit payment", "OWN_TRF": "Transfer between own accounts",
                     "MISC_CR": "Other credit", "MISC_DR": "Other debit", "FEE": "Bank fee"})
    add("TXN_TYPE_TAG", {"COLLECTION": "Collection from customer", "SUPPLIER": "Payment to supplier", "RENT": "Rent",
                         "SALARY": "Salaries", "UTILITY": "Utility bill", "GOV": "Government payment",
                         "FINANCING": "Financing", "CARD_SETTLEMENT": "Card acquiring settlement", "CARD": "Corporate card",
                         "CHEQUE": "Cheque", "OWN": "Own account", "FEE": "Fee", "OTHER": "Other"})
    add("ORIGIN_TYPE", {"PAYROLL": "Payroll file (FILE_ID)", "LOAN": "Financing contract", "STANDING_ORDER": "Standing order",
                        "SADAD": "SADAD bill", "LC": "Letter of credit", "DEPOSIT": "Deposit", "CORP_CARD": "Corporate card statement or top-up",
                        "CHEQUE": "Cheque", "CARD_SETTLEMENT": "Merchant settlement", "INVOICE": "Invoice issued by the client"})
    add("CHANNEL", {"ONLINE": "Online banking", "H2H": "Host to host", "BRANCH": "Branch", "POS": "Point of sale",
                    "SYSTEM": "System generated"})
    add("PRODUCT_CODE", {"CA01": "Current account", "CA02": "Payroll current account", "CI01": "Call investment",
                         "CM01": "Cash margin (restricted)"})
    add("SEGMENT", {"SME": "Small and medium enterprises", "MID_CORPORATE": "Mid corporate",
                    "LARGE_CORPORATE": "Large corporate"})
    add("SECTOR", {k: v[0] for k, v in SECTORS.items()})
    add("PARTY_TYPE", {"Business Customer": "Business customer"})
    add("PAYROLL_STA_CD", {"PAID": "Paid", "CANCELED": "Cancelled", "TEST": "Test file", "CNFRM_DPLCT": "Confirmed duplicate"})
    add("PAYROLL_FILE_TYPE", {"SALARY": "Monthly salaries", "BONUS": "Bonus", "WPS": "Wage protection system",
                              "NON_WPS": "Outside wage protection system"})
    add("FINANCING_STATUS", {"PAID": "Paid on time", "PAID_LATE": "Paid late", "OVERDUE": "Overdue", "NOT_DUE": "Not yet due"})
    return pd.DataFrame(rows)
