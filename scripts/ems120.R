pacman::p_load(readxl, data.table, tidyverse, lubridate, stringi, patchwork, scales, RColorBrewer) 

dir0 = "D:"
source(paste0(dir0, '/scripts/f/phe.f.R'))
replacement <- c('年龄', '性别', '呼救原因', '开始受理时刻', '派车时间', '去程时间', '现场时间', '返程时间', '急救时间') 
pattern <- c('年龄|病人年龄', '性别|病人性别', '^呼救原因|^呼叫原因', '^开始受理时刻|^开始时刻|^摘机时间', '^派车时间|^受理调度时间', '^去程时间|^去程在途时间', '^现场时间|^现场救援时间|^现场治疗时间|^现场急救时间', '^返程时间|^返程在途时间', '^急救时间|^急救反应时间')
# 派车时间 = 驶向现场时刻 - 开始受理时刻
# 去程时间 = 到达现场时刻 - 驶向现场时刻
# 现场时间 = 病人上车时刻 - 到达现场时刻
# 返程时间 = 到达医院时刻 - 病人上车时刻
# 急救时间 = ++++
dir.dat <- "D:/projects/01大学/02科研论文/ems120"
years <- 2013:2023
dxs.cn <- c("创伤-暴力事件", "创伤-交通事故", "创伤-跌倒", "理化中毒", "心脑血管疾病", "呼吸系统疾病", "内分泌系统疾病", "精神病", "创伤-高处坠落", "创伤-其他原因", "泌尿系统疾病", "消化系统疾病", "妇产科", "儿科", "其他-昏迷", "其他-其他症状", "其他-死亡")
dxs <- c("Violence", "Accident", "Fall", "Poisoning", "CVD", "Respiratory", "Endocrine", "Psychiatric", "Trauma.jump", "Trauma.other", "Urinary", "Digestive", "Ob/Gyn", "Pediatrics", "Coma", "Other", "Death")
dxs.vip <- dxs[1:8]; dxs.vip.color <- c("purple", "orange", "darkblue", "brown", "red", "green", "brown", "pink")
names(dxs.vip.color) <- dxs.vip; dxs.vip4 <- dxs.vip[5:8]; dxs.vip4.color <- dxs.vip.color[5:8]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 读入数据
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dat.list0 <- list()
for (year in years) {
	print(year)
	dat <- read_excel(paste0(dir.dat, '/120数据/清洗后数据/', year, '.xlsx')) 
	dat.list0[[as.character(year)]] <- dat
}
sum(map_int(dat.list0, nrow))
names(dat.list0)[map_lgl(dat.list0, ~ !"年龄" %in% names(.x))] # 没有年龄变量的
sum(map_int(dat.list0, ~ sum(sum(nchar(.x$联系电话) == 11, na.rm = TRUE)))) # 电话号码不是11位数的
sapply(dat.list0, function(daf) { daf %>% count(联系电话, sort = TRUE) %>% pull(n) %>% head(50) }) # 🏮
imap_dfr(dat.list0, ~{ daf <- .x
	cc <- daf %>% count(联系电话, name = "calls"); n_phones <- sum(cc$calls > 5); n_calls <- sum(cc$calls[cc$calls > 5])
	tibble(Year = as.integer(.y), repeat_phones = n_phones, repeat_calls = n_calls)
})

dat.list <- lapply(dat.list0, function(datin) {
	dat <- datin %>% filter(!is.na(疾病类型), nchar(联系电话) == 11) %>% 
	group_by(联系电话) %>% filter(n() <= 5) %>% ungroup() %>% # 去掉每年5次以上的
	mutate(across(where(is.POSIXct), ~ format(.x, "%Y-%m-%d %H:%M:%S"))) # 去掉时区 🏮
	dup_cols <- grep("^派车时间\\.\\.", names(dat), value = TRUE)
	if (length(dup_cols) == 2) { dat <- dat %>% rename(`派车时间.raw` = !!sym(dup_cols[1]), `派车时间` = !!sym(dup_cols[2])) }
	names(dat) <- stringi::stri_replace_all_regex(names(dat), pattern = pattern, replacement = replacement, vectorize_all = FALSE)
	for(col in c("接车地址经度", "接车地址纬度")) { if(! col %in% names(dat)) dat[[col]] <- NA }
	dat <- dat %>% dplyr::select(年龄, 性别, 联系电话, 疾病类型, 开始受理时刻, 派车时间, 去程时间, 现场时间, 返程时间, 急救时间, 接车地址经度, 接车地址纬度)
	dat <- dat %>% mutate(
		年龄 = as.numeric(年龄), 联系电话 = as.character(联系电话), 
		时刻 = as_datetime(开始受理时刻), 日期 = as.Date(时刻), 钟点 = format(时刻, "%H:%M:%S"), hour = hour(hms(钟点)),
		phone = substring(联系电话, 4, 11),
		疾病类型 = ifelse(疾病类型 %in% c("其他-胸闷", "神经系统疾病-脑卒中", "神经系统疾病-其他疾病", "心血管系统疾病-其他疾病", "心血管系统疾病-胸痛"), "CVD", 
				 疾病类型),
		疾病类型 = recode(疾病类型, !!!setNames(dxs, dxs.cn), .default = 疾病类型)
	) %>% group_by(疾病类型) %>% filter(n() >= 50) %>% ungroup() 
	for (n in 0:9) {dat[[paste0("phone_n", n)]] <- str_count(dat$phone, as.character(n))}
	dat <- dat %>% mutate(
		phone_sco = phone_n8 + phone_n9 *0.75 + phone_n6*0.5 + phone_n1 *0.25,
		# phone_sco = phone_n8 + phone_n9 + phone_n6 + phone_n1,
		# phone_sco = phone_n8 + phone_n9 *0.5 + phone_n6 *0.5 + phone_n1 *0.5
		phone_grp = factor(ifelse(phone_n4 >= 1, "low", ifelse(phone_sco <= quantile(phone_sco, 0.75), "middle", "high")), levels = c("low", "middle", "high"))
	)
	dat
})
sapply(dat.list, function(daf) table(daf$phone_sco))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 表1. 基本信息ℹ
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
show_row <- function(df) { df %>% slice(1) %>% mutate(across(everything(), as.character)) %>% pivot_longer(everything())}
sum(map_int(dat.list, nrow))
	sapply(dat.list, show_row, simplify = FALSE) # 🏂
	sapply(dat.list, function(daf) { daf %>% count(联系电话, sort = TRUE) %>% pull(n) %>% head(50) }) # 🏮
	sapply(dat.list, function(daf) table(daf$phone_grp))
	sapply(dat.list, function(daf) quantile(daf$phone_sco, 0.75)) # 🏮

the_table <- imap_dfr(dat.list, ~{ daf <- .x
	n <- nrow(daf); n_uniq <- n_distinct(daf$联系电话); fem_pct <- sum(daf$性别 == "女", na.rm = TRUE)/ n * 100
	high_n <- sum(daf$phone_grp == "high", na.rm = TRUE); high_pct<- high_n / n * 100
	low_n <- sum(daf$phone_grp == "low", na.rm = TRUE); low_pct <- low_n / n * 100
	tibble( Year = as.integer(.y), Age = sprintf("%.1f (%.1f)", mean(daf$年龄, na.rm = TRUE), sd(daf$年龄, na.rm = TRUE)),
		Female = sprintf("%.1f%%", fem_pct), 
		N = format(n, big.mark = ","), N_uniq_pct = sprintf("%.2f%%", 100 * n_uniq / n),
		Low = sprintf("%s (%.1f%%)", format(low_n, big.mark = ","), low_pct),
		High = sprintf("%s (%.1f%%)", format(high_n, big.mark = ","), high_pct) 
	)
})
the_table; fwrite(the_table, file = "table1.txt", sep = "\t", na = NA, row.names = FALSE, quote = FALSE)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 图S1. 📱疾病比例
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dat <- map_dfr(years, ~{
 	dat.list[[as.character(.x)]] %>% count(疾病类型, name = "count") %>% mutate(year = .x, pct = count/sum(count)) %>% select(year, 疾病类型, count, pct)
})
lev2024 <- dat %>% filter(year == 2024) %>% arrange(desc(pct)) %>% pull(疾病类型) # 以2024年的发病率排名
dat <- dat %>% mutate(疾病类型 = factor(疾病类型, levels = lev2024))

the_plot <- ggplot(dat, aes(factor(year), count, fill = 疾病类型)) +
	geom_col(position = "fill", color = "white") +
	geom_text(aes(label = ifelse(疾病类型 %in% lev2024[(length(lev2024)-3):length(lev2024)], NA_character_, sprintf("%.1f%%", pct * 100))), position = position_fill(vjust = 0.5), size = 3) +
	scale_fill_hue(name = "Category:") +
	scale_y_continuous(labels = NULL, expand = c(0, 0)) +
	labs(x = "Year", y = "Percentage") +
	theme_minimal(base_size = 12) + theme(axis.title = element_text(face = 'bold'), axis.text = element_text(face = 'bold'))
the_plot; ggsave("FigS1.png", the_plot, width = 8, height = 10, units = "in", dpi = 600)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 图1. 🛏疾病类型每周波动情况
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
years <- 2019:2024
weekly <- bind_rows( lapply(years,function(y) dat.list[[as.character(y)]] %>% 
	filter(疾病类型%in%dxs.vip) %>% mutate(week = week(日期)) %>% group_by(year = y,week,disease = 疾病类型) %>%
	summarise(call_count = n(),days = n_distinct(as.Date(日期)),.groups = "drop") %>% filter(days == 7) %>%
	mutate(week_start = as.Date(paste0(year,"-01-01")) + weeks(week-1))
))

plots <- lapply(seq_along(years),function(i){
	daf <- weekly %>% filter(year == years[i]) %>% mutate(call_capped = pmin(call_count,1000), over = call_count>1000)
	ggplot(daf,aes(week_start, call_capped, color = disease, linetype = disease, size = disease))+
	geom_line() + geom_text(data = daf %>% filter(over),aes(label = "*"), vjust = -0.5, show.legend = FALSE)+
	scale_color_manual(values = setNames(dxs.vip.color, dxs.vip)) +
	scale_linetype_manual(values = setNames(c(rep("dotted",4), rep("solid",4)), dxs.vip)) +
	scale_size_manual(values = setNames(c(rep(1.5, 4),rep(1, 4)), dxs.vip), guide = FALSE) +
	scale_x_date(breaks = date_breaks("3 months"), labels = date_format("%b",locale = "en")) +
	scale_y_continuous(limits = c(0,1000), breaks = seq(0, 1000, 250))+
	labs(title = years[i], x = NULL, y = if(i %in% c(1,4)) "Number of Calls" else NULL)+
	theme_minimal(base_size = 11) + 
	theme( axis.title = element_text(face = 'bold'), axis.text = element_text(face = 'bold'), 
		axis.line = element_line()
	)
})
the_plot <- wrap_plots(plots, nrow = 2, ncol = 3, guides = "collect") &
	scale_color_manual(name = NULL, breaks = dxs.vip, values = setNames(dxs.vip.color, dxs.vip)) &
	scale_linetype_manual(name = NULL, breaks = dxs.vip, values = setNames(c(rep("dotted", 4), rep("solid", 4)), dxs.vip)) &
	theme(legend.position = "bottom", legend.text = element_text(face = "bold", size = 14)) &
	guides(color = guide_legend( nrow = 1, byrow = TRUE, override.aes = list(size = 6, stroke = 1.5, shape = 18)),
		linetype = guide_legend( nrow = 1, byrow = TRUE, override.aes = list(linewidth = 2, shape = 2 ))
	)
the_plot; ggsave("Fig1.png", the_plot, width = 9, height = 6, dpi = 300)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 图2. 幸运者的🎇发病相对比例
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
years <- 2013:2024
dat <- lapply(years, function(y) {
	daf <- dat.list[[as.character(y)]]
	ph <- prop.table(table(daf$phone_grp))
	dxp <- prop.table(table(daf$疾病类型, daf$phone_grp), 1)
	data.frame(year = y, disease = dxs.vip, ph_high = sweep(dxp, 2, as.numeric(ph), "/")[dxs.vip, "high"], ph_low = sweep(dxp, 2, as.numeric(ph), "/")[dxs.vip, "low"])
}) %>% bind_rows()

cols <- dxs.vip.color; names(cols) <- dxs.vip
plots <- lapply(seq_along(dxs.vip), function(i) {
	daf <- subset(dat, disease == dxs.vip[i]) %>% 
	mutate( ph_low_disp = pmax(ph_low, 0.9), ph_high_disp = pmin(ph_high, 1.1), low_flag = ph_low < 0.9, high_flag = ph_high > 1.1)
	sy <- i %% 2 == 1
	ggplot(daf, aes(color = disease)) +
	geom_segment(aes(x = 1, xend = ph_low_disp, y = year, yend = year), linetype = "dashed", color = "grey80") +
	geom_segment(aes(x = 1, xend = ph_high_disp, y = year, yend = year), linetype = "dashed") +
	geom_point(aes(x = ph_low_disp, y = year), color = "grey50", size = 3) +
	geom_point(aes(x = ph_high_disp, y = year), size = 3) +
	geom_text(data = subset(daf, low_flag), aes(x = ph_low_disp, y = year), label = "<", hjust = 1.2) +
	geom_text(data = subset(daf, high_flag), aes(x = ph_high_disp, y = year), label = ">", hjust = 0) +
	geom_vline(xintercept = 1, color = "black") +
	scale_color_manual(name = "", values = cols) +
	scale_x_continuous(limits = c(0.9, 1.1)) +
	scale_y_continuous(breaks = years, labels = years) +
	labs( title = dxs.vip[i], x = if (i %in% 5:6) "Relative Risk" else NULL, y = if (sy) "Year" else NULL) +
	theme_minimal() +
	theme( axis.text = element_text(face = 'bold'), axis.title = element_text(face = 'bold'), axis.line = element_line(), legend.position = NULL )
})
the_plot <- wrap_plots(plots, nrow = 4, ncol = 2, guides = "collect") & theme(legend.position = "bottom", legend.text = element_text(face = "bold", size = 12)) 
the_plot; ggsave("Fig2.png", the_plot, width = 11.2, height = 10, dpi = 600)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 图3. 幸运者的急救🚑时间
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pacman::p_load(zoo, broom, forcats, circlize)
vars <- c("派车时间", "去程时间", "现场时间"); vars.en <- c("Dispatch", "Driving", "Onsite")
high.colors <- c("blue", "purple", "red")
years <- 2013:2024
probs <- c(0, 0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99, 1)
yearly_time <- lapply(vars, function(var) {
	map_dfr(years, function(y) { 
		daf <- dat.list[[as.character(y)]] # dat.list.capped[[as.character(y)]]
		tibble( Variable = var, Year = y, LowMean = mean(daf[[var]][daf$phone_grp == "low"]/60, na.rm = TRUE), HighMean = mean(daf[[var]][daf$phone_grp == "high"]/60, na.rm = TRUE))})
}) %>% bind_rows()

dxs.list <- dxs.vip[c(1,3,4)]
hourly_frq <- map_dfr(years, function(y) {
	dat.list[[ as.character(y) ]] %>% filter(疾病类型 %in% dxs.list, phone_grp %in% c("low","high")) %>% select(疾病类型, phone_grp, hour, 现场时间)
	}) %>% group_by(疾病类型, phone_grp, hour) %>% summarise(mean_time = mean(现场时间, na.rm = TRUE), .groups = "drop") %>%
	pivot_wider(names_from = phone_grp, values_from = mean_time, values_fill = list(low = 0, high = 0)) %>% arrange(疾病类型, hour)

plots <- lapply(seq_along(vars), function(i) {
	var_i <- vars[i]
	var_data <- filter(yearly_time, Variable == var_i)
	gm <- mean(var_data$HighMean, na.rm = TRUE)
	ggplot(var_data, aes(y = Year)) +
	geom_segment(aes(x = LowMean, xend = HighMean, yend = Year), linetype = "dashed", color = "grey70") +
	geom_point(aes(x = LowMean), color = "grey50", size = 3) + geom_point(aes(x = HighMean), color = high.colors[i], size = 3, shape = 17) +
	geom_vline(xintercept = gm, color = high.colors[i], linetype = "dashed") + 
	labs(title = vars.en[i], x = "Time (mins)", y = if(i == 1) "Year" else NULL) +
	scale_y_continuous(breaks = years, labels = if(i == 1)years else NULL) + 
	theme_minimal(base_size = 12) + theme( axis.title = element_text(face = 'bold'), axis.text = element_text(face = 'bold'), axis.line = element_line())	
})
the_plot <- wrap_plots(plots, nrow = 1, ncol = 3)
the_plot; ggsave("Fig3a.png", the_plot, device = "png", width = 10, height = 6, units = "in", dpi = 600)

make_hourly_circle <- function(hourly_frq, dxs.array, high.colors, bg.colors) {
	circos.clear(); circos.par(start.degree = 90, gap.degree = 0)
	circos.initialize(factors = "all", xlim = c(0,24))
	for(i in seq_along(dxs.array)) {
		dx <- dxs.array[i]; datmp <- filter(hourly_frq, 疾病类型 == dx) %>% arrange(hour) # 必须要🏮
		circos.trackPlotRegion( 
			factors = "all", track.index = i, ylim = range(datmp$low, datmp$high), bg.col = bg.colors[i], bg.border = NA, track.height = 0.15,
			panel.fun = function(...) {circos.lines(datmp$hour, datmp$low, col = "darkgray", lwd = 2); circos.lines(datmp$hour, datmp$high, col = high.colors[i], lwd = 2)}
		)
	}
	circos.axis(h = "top", major.at = 0:23, labels = sprintf("%02d", 0:23), labels.cex = 1.2, minor.ticks = 0, sector.index = "all", track.index = 1)
	for(i in seq_along(dxs.array)) {circos.text(x = -0.1, y = get.cell.meta.data("ylim", track.index = i)[2] - 88, labels = dxs.array[i], facing = "inside", adj = c(1,0.5), cex = 1.1, font = 2, sector.index = "all", track.index = i)}
}

high.colors <- dxs.vip4.color
bg.colors <- c("#FDE0DD", "#E0F3DB", "#D9EDF7", "yellow")
make_hourly_circle(hourly_frq, dxs.list, high.colors, bg.colors)


# Subset dat.list for years 2022 and 2023
dat.list2 <- dat.list[c("2022", "2023")]
saveRDS(dat.list2, "dat.list.rds")
dat.lis <- readRDS("dat.list.rds") 


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 图4. 幸运者的疫情管控🛑影响
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
day1 <- as.Date("2022-03-14"); day2 <- as.Date("2022-03-20")
did_simple = TRUE # DID: difference in difference

dat <- dat.list[["2022"]] %>% mutate(疾病类型 = factor(疾病类型, dxs.vip4)) %>%
	filter(疾病类型 %in% dxs.vip4, phone_grp != "middle", between(日期, day1 - 7, day2 + 7))
	daily_cnt <- dat %>% count(疾病类型, phone_grp, 日期, name = "count")

if (did_simple) {
	did_glm <- function(df, start_date, end_date){
	sub <- df %>% filter(between(日期, start_date, end_date))
	fit <- glm(count ~ phone_grp, family = poisson, data = sub)
	td <- broom::tidy(fit) %>% filter(term == "phone_grphigh")
	td %>% transmute(OR = exp(estimate), lo = exp(estimate - 1.96*std.error), hi = exp(estimate + 1.96*std.error), p.value = p.value)
	}
} else {
	did_glm <- function(df, pre_start, pre_end, post_start, post_end) {
	df_sub <- df %>% filter(between(日期, pre_start, post_end)) %>%
		mutate( period = if_else(日期 >= post_start & 日期 <= post_end, "post", "pre"), period = factor(period, c("pre","post")))
		td <- glm(count ~ period * phone_grp, family = poisson, data = df_sub) %>% broom::tidy() %>% filter(term == "periodpost:phone_grphigh") 
		tibble( OR = exp(td$estimate), lo = exp(td$estimate - 1.96*td$std.error), hi = exp(td$estimate + 1.96*td$std.error), p.value = td$p.value)
	}
}
did_calc <- function(daily_cnt, start_date, end_date, dxs) {
	daily_cnt %>% group_by(疾病类型) %>% nest() %>%
	#map(data, ~ did_glm(.x, pre_start = day1 - 7, pre_end = day1 - 1, post_start = day1, post_end = day2)))) %>% unnest(res) %>% 
	mutate(res = map(data, ~ did_glm(.x, start_date, end_date))) %>% unnest(res) %>%
	mutate(sig = case_when( p.value < .005 ~ "**", p.value < .05 ~ "*", TRUE ~ ""), 疾病类型 = fct_relevel(疾病类型, dxs)) %>% ungroup()
}
did.pre <- did_calc(daily_cnt, day1, day2, dxs.vip4)
did.post <- did_calc(daily_cnt, day2, day2 + 7, dxs.vip4)

daily_plot <- function(title_txt, daf, day1, day2, dxs.color) {
	ggplot(daf, aes(日期, count)) + geom_vline(xintercept = c(day1, day2),
	linetype = "dashed", color = "orange", size = 1) + geom_line(data = filter(daf, phone_grp == "low"),
	aes(group = 1), color = "darkgray", size = 1) + geom_line(data = filter(daf, phone_grp == "high"),
	aes(color = 疾病类型), size = 1) + scale_color_manual(values = dxs.color) + facet_wrap(~疾病类型, scales = "free_y", ncol = 1) +
	scale_x_date(labels = date_format("%b %d", locale = "en")) +
	scale_y_continuous(breaks = scales::pretty_breaks(n = 2), labels = scales::label_number(accuracy = 1)) +
	labs(title = title_txt, x = NULL, y = NULL) +
	theme_minimal(base_size = 12) + theme(axis.text = element_text(face = 'bold'), legend.position = "none")
}

did_plot <- function(title_text, daf, dxs, dxs.color) { ggplot(daf, aes(
	x = OR, y = factor(疾病类型, levels = rev(dxs)), color = 疾病类型)) + # rev将顺序变成从上到下
	geom_vline(xintercept = 1, linetype = "dashed") +
	geom_point(size = 3) + geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.2) +
	geom_text(aes(label = sig), hjust = -0.5, vjust = 0.5) +
	scale_color_manual(values = dxs.color) +
	labs(title = title_text, x = "Rate Ratio (high vs low)", y = NULL) +
	theme_minimal(base_size = 12) + 
	theme(axis.text = element_text(face = 'bold'), legend.position = "none", plot.margin = margin(t = 5, r = 20, b = 5, l = 5))
}

p1 <- daily_plot("A. Daily Calls (March 07 to March 21)", daily_cnt, day1, day2, dxs.vip4.color)
p2 <- did_plot("B. DID during PHSM (Mar 14–20)", did.pre, dxs.vip4, dxs.vip4.color)
p3 <- did_plot("C. DID after PHSM (Mar 21–27)", did.post, dxs.vip4, dxs.vip4.color)
the_plot <- (p1 / plot_spacer() / (p2 | p3)) + plot_layout(heights = c(3,0.1,1), widths = c(2,1))
the_plot; ggsave("Fig4.png", the_plot, device = "png", width = 10, height = 12, units = "in", dpi = 600)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 图5. 幸运者的疫情放开影响🎇
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
day1 <- as.Date("2022-11-11"); day2 <- as.Date("2022-12-07")
dat.1 = dat.list[["2022"]] %>% filter(between(日期, day1 - 10, as.Date("2022-12-31"))) %>% select(疾病类型, 日期, phone_grp)
dat.2 = dat.list[["2023"]] %>% filter(between(日期, as.Date("2023-01-01"), day2 + 24)) %>% select(疾病类型, 日期, phone_grp)
dat <- rbind(dat.1, dat.2) %>% mutate(疾病类型 = factor(疾病类型, dxs.vip)) %>%
	filter(疾病类型 %in% dxs.vip, phone_grp != "middle")

daily_cnt <- dat %>% count(疾病类型, phone_grp, 日期, name = "count")
did.pre <- did_calc(daily_cnt, day1, day2, dxs.vip)
did.post <- did_calc(daily_cnt, day2, day2 + 7, dxs.vip)

p1 <- daily_plot("A. Daily Calls (last two monthes of 2022)", daily_cnt, day1, day2, dxs.vip.color)
p2 <- did_plot("B. DID of the frist open-up", did.pre, dxs.vip, dxs.vip.color)
p3 <- did_plot("C. DID of the final open-up", did.post, dxs.vip, dxs.vip.color)
the_plot <- (p1 | (p2 / p3)) + plot_layout(widths = c(2, 1))
the_plot; ggsave("Fig5.png", the_plot, device = "png", width = 8, height = 8, units = "in", dpi = 600)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 图6. 幸运者的房价🏠
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pacman::p_load(sf) 
house <- read_excel(paste0(dir.dat,'/120数据/深圳房价.xlsx')) %>% # 🏠
	mutate(house.id = 1:n()) %>% rename(house.price = 房价) %>% select(-小区, -地址)
	house_sf <- st_as_sf(house, coords = c("Lon", "Lat"), crs = 4326) %>% # 4326是经纬度
	st_transform(house_sf, crs = 3857) %>% mutate(geometry.house = st_geometry(.)) # 3857是meter
	house_buffer <- st_buffer(house_sf, dist = 1000) # 方圆1千米范围内
X <- dat.list[["2021"]] %>% select(phone_sco, phone_grp, 疾病类型, 接车地址经度, 接车地址纬度) %>% 
	mutate(X.id = 1:n())
	X.sf <- st_as_sf(X, coords = c("接车地址经度", "接车地址纬度"), crs = 4326) %>% st_transform(., crs = 3857) # %>% mutate(geometry.X = st_geometry(.))
dat0 <- st_intersection(house_buffer, X.sf) # 合并后的 geometry 来自第一个变量
dat0 <- dat0 %>% group_by(X.id) %>% # 一个人只属于一个house
	mutate(distance = st_distance(geometry.house, geometry, by_element = TRUE)) %>% 
	slice_min(order_by = distance) %>% # 离TA最近的那个house
	st_drop_geometry(.) %>% ungroup() %>% rename(geometry = geometry.house) # 不再需要打电话人的地址了
	saveRDS(dat, "120.rds")
	summary(lm(house.price ~ phone_sco, data = dat0))

dat <- dat0 %>% group_by(house.id) %>%
	summarise(house.price = first(house.price), geometry = first(geometry), phone_sco.mean = round(mean(phone_sco, na.rm = TRUE), 2), .groups = "drop") %>% 
	st_as_sf() %>% st_transform(crs = 4326) %>% 
	mutate(lon = st_coordinates(geometry)[,1], lat = st_coordinates(geometry)[,2]) %>% st_drop_geometry(.)
	fwrite(dat, file = "D:/files/120.txt", append = FALSE, sep = "\t", row.names = FALSE, quote = FALSE)
	dat$house.price <- log10(dat$house.price); dat$X <- dat$phone_sco.mean
	
par(mar = c(5, 4, 4, 5) + 0.1, font.lab = 2, font.axis = 2) 
	myhist <- hist(dat$house.price, freq = TRUE, main = "", breaks = 10, xlim = c(3,6), xlab = "Housing price", ylab = "")
	X.avgs <- by(dat$X, cut(dat$house.price, breaks = myhist$breaks), function(x) mean(x, na.rm = TRUE))
	X.sds <- by(dat$X, cut(dat$house.price, breaks = myhist$breaks), function(x) sd(x, na.rm = TRUE)) 
	par(new = T)
	plot(myhist$mids, X.avgs, xlim = range(myhist$breaks), ylim = c(1,3), pch = 16, axes = FALSE, xlab = NA, ylab = NA, cex = 1.2, col = "blue")
	arrows(myhist$mids, X.avgs-X.sds, myhist$mids, X.avgs+X.sds, angle = 90, code = 3, length = 0.05, col = "darkgray")
	axis(side = 4); mtext(side = 4, line = 3, "Phone score (mean)", col = "blue")

the_plot <- recordPlot()
png("Fig6a.png", width = 8, height = 4, units = "in", res = 300); replayPlot(the_plot); dev.off()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 图S2. 敏感性分析
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dat.list2 <- map(
	dat.list, ~ .x %>% mutate(
		phone_sco.A = phone_n8 + phone_n9 *0.75 + phone_n6 *0.5 + phone_n1 *0.25,
		phone_sco.B = phone_n8 + phone_n9 + phone_n6 + phone_n1,
		phone_sco.C = phone_n8 + phone_n9 *0.5 + phone_n6 *0.5 + phone_n1 *0.5
	) %>% mutate(across(starts_with("phone_sco."), list(
		q65 = ~ factor(ifelse(phone_n4 >= 1, "low", ifelse(. <= quantile(., .65), "middle", "high")), levels = c("low","middle","high")),
		q75 = ~ factor(ifelse(phone_n4 >= 1, "low", ifelse(. <= quantile(., .75), "middle", "high")), levels = c("low","middle","high")),
		q85 = ~ factor(ifelse(phone_n4 >= 1, "low", ifelse(. <= quantile(., .85), "middle", "high")), levels = c("low","middle","high"))), .names = "{.col}_{.fn}")
	)
)

df_all <- imap_dfr( dat.list2, ~ .x %>% pivot_longer(
	cols = matches("^phone_sco\\.[A-Z]+_q\\d+$"), names_to = c("score","quant"),
	names_pattern = "phone_sco\\.([A-Z]+)_(q\\d+)", values_to = "grp"
	) %>% count(year = .y, score, quant, grp) %>%
	group_by(year, score, quant) %>% mutate(pct = n / sum(n)) %>% ungroup()
)
wide <- df_all %>% filter(grp %in% c("low","high")) %>% select(year, score, quant, grp, pct) %>%
	pivot_wider(names_from = grp, values_from = pct) %>% group_by(score, quant) %>%
	mutate(k = max(low, na.rm = TRUE) / max(high, na.rm = TRUE), high_scaled = high * k) %>%
	ungroup() %>% mutate(year = as.integer(year), label = paste0(score, ".", quant))
panel_labels <- wide %>% distinct(label) %>% pull(label)
high.colors2 <- setNames(hue_pal()(length(panel_labels)), panel_labels)

plots <- map(panel_labels, function(lbl) {
	dfp <- filter(wide, label == lbl) %>% mutate(year = as.integer(year))
	y.lim <- c(0.425, 0.475) # range(c(dfp$low, dfp$high_scaled), na.rm = TRUE)
	span <- diff(y.lim) * 0.2; ylim20p <- y.lim + c(-span, +span)
	ggplot(dfp, aes(x = year)) + geom_line(aes(y = low), color = "darkgray", size = 1) +
	geom_point(aes(y = low), color = "darkgray", size = 3, shape = 21, fill = "white") +
	geom_line(aes(y = high_scaled), color = high.colors2[lbl], size = 1) +
	geom_point(aes(y = high_scaled), color = high.colors2[lbl], size = 3, shape = 21, fill = "white") +
	scale_x_continuous(breaks = 2013:2024, limits = c(2013, 2024)) +
	coord_cartesian(ylim = ylim20p) + labs(y = lbl, x = NULL) +
	theme_minimal(base_size = 12) +
	theme( panel.grid.major = element_line(color = "gray80"), panel.grid.minor = element_blank(), 
	axis.title.x = element_blank(), axis.text = element_blank(), axis.ticks = element_blank(),
	axis.title.y = element_text(angle = 0, vjust = 0.5, face = "bold"), legend.position = "none"
	)
})
plots[[length(plots)]] <- plots[[length(plots)]] + theme(axis.text.x = element_text(face = 'bold'), axis.ticks.x = element_line()) + labs(x = "Year")
the_plot <- wrap_plots(plots, ncol = 1)
the_plot; ggsave("FigS2.png", the_plot, device = "png", width = 8, height = 8, units = "in", dpi = 600)
