tables = generate(n_clients=N_CLIENTS, last_month=SIM_END, seed=SEED)
cutoff = load_warehouse(tables, AS_OF, CONNECTION)
print(f"Warehouse as of {AS_OF}: cut-off day {cutoff} (project variable cutoff_override)")
