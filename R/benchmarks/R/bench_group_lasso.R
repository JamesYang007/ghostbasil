source('group_lasso_utils.R')

n <- 100
ps <- c(100, 200, 400, 800, 1600, 3200)
times <- 1

write.csv.default(n, 'n.csv')
write.csv.default(ps, 'p.csv')

bench.times <- bench(n, ps, n.groups.prop=0.2, n.lmdas=100, times=times, seed=9183)$times
write.csv.default(bench.times, 'group_lasso_low_groups_times.csv')

bench.times <- bench(n, ps, n.groups.prop=0.8, n.lmdas=100, times=times, seed=9183)$times
write.csv.default(bench.times, 'group_lasso_high_groups_times.csv')
