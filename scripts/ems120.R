dir0 <- ifelse(Sys.info()[["sysname"]] == "Windows", "D:/", "/work/sph-huangj")
source(file.path(dir0, "scripts", "f", "0conf_ML.R"))
setwd(file.path(dir0, "analysis", "ems120"))
pacman::p_load(readxl, writexl, data.table, tidyverse, scales, RColorBrewer, reticulate, lubridate, patchwork, zoo, broom, forcats, circlize) 
invisible(lapply(c("phe.f.R", "plot.f.R"), \(f) source(file.path(dir0, "scripts", "f", f))))
dir.dat <- "D:/data/ems120"
years <- 2013:2024
vars.basic <- c("电话", "地址", "地址类型", "开始受理时刻", "派车时间", "去程时间", "现场时间", "返程时间", "急救时间", "疾病类型", "接车地址经度", "接车地址纬度")
vars.basic.alias <- c("^病人电话号码|^联系电话.1|^联系电话", "^接车地址$|^接车地点$|^现场地址$", "地址类型",
	"^开始受理时刻|^开始时刻|^摘机时刻|^收到指令时刻", "^派车时间|^受理调度时间", "^去程时间|^去程在途时间",
	"^现场时间|^现场救援时间|^现场治疗时间|^现场急救时间", "^返程时间|^返程在途时间", "^急救时间|^急救反应时间", "疾病类型", "接车地址经度", "接车地址纬度")
vars.dxs <- c("性别", "年龄", "呼救原因", "病种判断", "病因", "伤病程度", "症状", "主诉", "病史", "初步诊断", "补充诊断")
vars.dxs.alias <- c("^性别$|病人性别|患者性别", "^年龄$|病人年龄|患者年龄", "^呼救原因|^呼叫原因", "病种判断", "^病因$|辅助诊断",
	"伤病程度|病情分级", "^症状$|患者症状", "^主诉$|病情\\(主诉\\)", "^病史$|现病史", "^初步诊断$", "初步诊断2|补充诊断")
vars <- c(vars.basic, vars.dxs); vars.alias <- c(vars.basic.alias, vars.dxs.alias)
vars.time <- c("开始受理时刻", "驶向现场时刻", "到达现场时刻", "病人上车时刻", "到达医院时刻")
dxs <- list(
	"Traffic" = "创伤-交通事故", "Poison" = "理化中毒",
	"Trauma" = c("创伤-暴力事件", "创伤-跌倒", "创伤-高处坠落", "创伤-其他原因", "其他-昏迷"),
	"CVD" = c("其他-胸闷", "神经系统疾病-脑卒中", "神经系统疾病-其他疾病", "心血管系统疾病-其他疾病", "心血管系统疾病-胸痛"),
	"Respiratory" = "呼吸系统疾病", "Mental" = "精神病", "NCD-Other" = c("泌尿系统疾病", "消化系统疾病", "内分泌系统疾病"),
	"Other" = c("妇产科", "儿科", "其他-其他症状"), "Death" = "其他-死亡"
)
dxs.raw <- unlist(dxs, use.names = FALSE)
dxs.grp <- setdiff(names(dxs), "Other") 
dxs.grp.color <- setNames(rainbow(length(dxs.grp), s = 0.8, v = 0.85), dxs.grp)
map_grp <- stack(dxs) %>% setNames(c("dx_raw", "dx_grp"))
grp_use <- c("low", "high")
roll3 <- function(x) zoo::rollmean(x, 3, fill = NA, align = "center")
sig_star <- function(p) dplyr::case_when(p < .0001 ~ "***", p < .001 ~ "**", p < .01 ~ "*", TRUE ~ "")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 🚩 读入数据
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if (file.exists("dat0.list.rds")) dat0.list <- readRDS("dat0.list.rds")

dat0.list <- list()
for (year in years) {
	cat(sprintf("========== %d ==========\n", year))
	dat <- read_excel(file.path(dir.dat, "120数据", "清洗后数据", paste0(year, ".xlsx"))); names(dat) <- trimws(names(dat))
	col_names <- names(dat); new_names <- col_names; missing_vars <- c()
	for (i in seq_along(vars)) {
		all_m <- grep(vars.alias[i], col_names, value=TRUE)
		if (!length(all_m)) missing_vars <- c(missing_vars, vars[i]) else {
			best_m <- all_m[1]
			if (length(all_m) > 1) for (p in strsplit(vars.alias[i], "\\|")[[1]]) { m <- grep(p, all_m, value=TRUE); if (length(m)) { best_m <- m[1]; break } }
			new_names[match(best_m, col_names)] <- vars[i]
		}
	}
	if (length(missing_vars)) cat(sprintf("-> 警告：%d年缺 '%s'\n", year, paste(missing_vars, collapse="' 和 '")))
	names(dat) <- new_names
	dat <- dat %>% filter(!is.na(电话), !is.na(性别), !if_all(c(主诉, 病史, 初步诊断, 补充诊断), is.na)) %>% mutate( # 🏮
		across(any_of(vars.time), \(x) ymd_hms(trimws(x))),
		派车时间 = if ("派车时间" %in% names(.)) as.numeric(派车时间) else if (all(c("驶向现场时刻","开始受理时刻") %in% names(.))) as.numeric(驶向现场时刻 - 开始受理时刻, units="secs") else NA_real_,
		去程时间 = if ("去程时间" %in% names(.)) as.numeric(去程时间) else if (all(c("到达现场时刻","驶向现场时刻") %in% names(.))) as.numeric(到达现场时刻 - 驶向现场时刻, units="secs") else NA_real_,
		现场时间 = if ("现场时间" %in% names(.)) as.numeric(现场时间) else if (all(c("病人上车时刻","到达现场时刻") %in% names(.))) as.numeric(病人上车时刻 - 到达现场时刻, units="secs") else NA_real_,
		返程时间 = if ("返程时间" %in% names(.)) as.numeric(返程时间) else if (all(c("到达医院时刻","病人上车时刻") %in% names(.))) as.numeric(到达医院时刻 - 病人上车时刻, units="secs") else NA_real_,
		急救时间 = if ("急救时间" %in% names(.)) as.numeric(急救时间) else if (all(c("派车时间","去程时间","现场时间","返程时间") %in% names(.))) 派车时间 + 去程时间 + 现场时间 + 返程时间 else NA_real_
	) %>% select(any_of(vars)) %>% mutate(
		电话 = as.character(sub("^0+", "", 电话)), 电话 = ifelse(nchar(电话) == 11, 电话, NA_character_),
		phone = ifelse(is.na(电话), NA_character_, substring(电话, 4, 11)),
		年龄 = as.numeric(gsub("岁$", "", 年龄)), 时刻 = 开始受理时刻, 日期 = as.Date(时刻), hour = hour(时刻)
	) %>% select(-时刻)
	dat0.list[[as.character(year)]] <- dat
}
saveRDS(dat0.list, "dat0.list.rds")
write_xlsx(dat0.list[["2019"]][1:10000, intersect(c(vars.dxs, "疾病类型"), names(dat0.list[["2019"]])), drop=FALSE], "2019.train_dx.xlsx")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 🚩 Dx和Phone机器学习🩺
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if (file.exists("dat.list.rds")) dat.list <- readRDS("dat.list.rds")
if (file.exists("dat1.list.rds")) dat1.list <- readRDS("dat1.list.rds")

reticulate::source_python("D:/scripts/main/ems120.py")
dat.list <- list()
for (year in years) {
	cat("Processing year:", year, "\n")
	key <- as.character(year); outfile <- paste0(key, ".xlsx")
	if (file.exists(outfile)) { dat.list[[key]] <- read_excel(outfile); next }
	res <- tryCatch({
		dat <- dat0.list[[key]]
		py_phone <- eval_phone_batch(dat$phone) # 🐂🐎
		dat <- dat %>% mutate(phone.sco = as.numeric(unlist(py_phone$sco)), phone.sco.reason = as.character(unlist(py_phone$reason)))
		dat_py <- dat %>% select(any_of(vars.dxs)) # 不能把datetime列传给python
		py_dx <- eval_dx_batch(dat_py, data_name = key) # 🐂🐎
		dat <- dat %>% mutate(
			疾病分类.关键词 = as.character(unlist(py_dx$kw)), 疾病分类.关键词.理由 = as.character(unlist(py_dx$kw_reason)),
			疾病分类.ML = as.character(unlist(py_dx$ml)), 疾病分类.ML.理由 = as.character(unlist(py_dx$ml_reason))
		)
		write_xlsx(dat, outfile); dat
	}, error = \(e) { cat("ERROR at year", key, "\n"); print(e); NULL })
	dat.list[[key]] <- res; rm(res); gc()
}
saveRDS(dat.list, "dat.list.rds")
lapply(dat.list, names)
for (year in 2014:2021) {
	tab <- table(dat.list[[as.character(year)]]$疾病类型, dat.list[[as.character(year)]]$疾病分类.ML, useNA = "no") # print(tab)
	concordance <- sum(diag(tab)) / sum(tab)
	cat("Year:", year, " | Concordance:", sprintf("%.1f%%", concordance * 100), "\n\n")
}

dat1.list <- list()
for (year in years) {
	year <- as.character(year); dat1 <- dat.list[[year]]
	if (is.null(dat1) || nrow(dat1)==0) { dat1.list[[year]] <- NULL; next }
	dat1.list[[year]] <- dat1 %>% add_count(电话, name="n_tel") %>% add_count(疾病分类.关键词, name="n_kw") %>% add_count(疾病分类.ML, name="n_ml") %>%
		mutate(电话 = ifelse(!is.na(电话) & n_tel > 5, NA_character_, 电话),
			疾病分类.关键词 = ifelse(!is.na(疾病分类.关键词) & n_kw < 50, NA_character_, trimws(疾病分类.关键词)),
			疾病分类.ML = ifelse(!is.na(疾病分类.ML) & n_ml < 50, NA_character_, trimws(疾病分类.ML)),
			phone.luck = case_when(is.na(phone.sco) ~ NA_character_, phone.sco <= 2 ~ "low", phone.sco <= 7 ~ "middle", phone.sco <= 10 ~ "high", TRUE ~ NA_character_),
			phone.luck = factor(phone.luck, levels=c("low", "middle", "high")),
			dx_raw = trimws(疾病分类.ML)
		) %>% left_join(map_grp, by = "dx_raw") %>% select(-n_tel, -n_kw, -n_ml)
}
saveRDS(dat1.list, "dat1.list.rds")
lapply(dat1.list, function(dat) table(dat$dx_grp, useNA = "ifany"))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 🚩 表1 🦋
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
the_table <- imap_dfr(dat1.list, ~{
	dat1 <- .x
	if (is.null(dat1) || nrow(dat1) == 0) return(tibble())
	n <- nrow(dat1); n_uniq <- n_distinct(dat1$电话, na.rm = TRUE)
	fem_n <- sum(dat1$性别 == "女", na.rm = TRUE); fem_pct <- fem_n / n * 100
	low_n <- sum(dat1$phone.luck == "low", na.rm = TRUE); low_pct <- low_n / n * 100
	high_n <- sum(dat1$phone.luck == "high", na.rm = TRUE); high_pct <- high_n / n * 100
	tibble(
		Year = as.integer(.y),
		N = format(n, big.mark = ","), N_uniq = format(n_uniq, big.mark = ","), N_uniq_pct = sprintf("%.2f%%", 100 * n_uniq / n),
		Age = sprintf("%.1f (%.1f)", mean(dat1$年龄, na.rm = TRUE), sd(dat1$年龄, na.rm = TRUE)),
		Female = sprintf("%s (%.1f%%)", format(fem_n, big.mark = ","), fem_pct),
		Low = sprintf("%s (%.1f%%)", format(low_n, big.mark = ","), low_pct),
		High = sprintf("%s (%.1f%%)", format(high_n, big.mark = ","), high_pct)
	)
})
the_table2 <- the_table %>% rename(
    `Year` = Year, `EMS calls, N` = N, `Unique caller numbers, N` = N_uniq, `Unique / total, %` = N_uniq_pct,
    `Age, mean (SD)` = Age, `Female, n (%)` = Female, `Low-luck, n (%)` = Low, `High-luck, n (%)` = High
)
the_table2
write_xlsx(the_table2, path = "Table1.xlsx")
data.table::fwrite(the_table2, file = "Table1.txt", sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 🚩 图1. 🛏疾病类型每周波动情况
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plot_dx_trend <- function(dat_list, yrs = 2017:2024, dx_var = "dx_grp", dxs = dxs.grp,
	time.unit = "weekly", y.unit = "auto", var.group = "dxs",
	cap = 1000, out_png = NA, width = 12, height = 9, dpi = 300) {
	time.unit <- match.arg(time.unit, c("weekly", "hourly"))
	y.unit <- if (y.unit == "auto") ifelse(time.unit == "weekly", "count", "pct") else match.arg(y.unit, c("count", "pct"))
	var.group <- match.arg(var.group, c("dxs", "years"))
	dat1 <- map_dfr(yrs, \(y){
		d <- dat_list[[as.character(y)]]
		if (is.null(d) || nrow(d) == 0) return(tibble())

		if (time.unit == "weekly") {
			d %>% filter(!is.na(.data[[dx_var]]), !is.na(日期), .data[[dx_var]] %in% dxs) %>%
				transmute(year = y, 日期 = as.Date(日期), dx = factor(as.character(.data[[dx_var]]), levels = dxs)) %>%
				mutate(week_start = floor_date(日期, "week", week_start = 1), week = isoweek(日期)) %>%
				group_by(year, week, dx, week_start) %>% summarise(call_count = n(), days = n_distinct(日期), .groups = "drop") %>% filter(days == 7)
		} else {
			d %>% filter(!is.na(.data[[dx_var]]), !is.na(hour), .data[[dx_var]] %in% dxs, between(hour, 0, 23)) %>%
				transmute(year = y, hour = as.integer(hour), dx = factor(as.character(.data[[dx_var]]), levels = dxs)) %>%
				count(year, dx, hour, name = "call_count") %>% complete(year, dx, hour = 0:23, fill = list(call_count = 0)) %>%
				group_by(year, dx) %>% mutate(pct = call_count / sum(call_count)) %>% ungroup()
		}
	})
	if (nrow(dat1) == 0) stop("🛑 Fig1: No data after filtering.")
	if (var.group == "dxs") {
		plots <- lapply(seq_along(yrs), \(i){
			dat2 <- dat1 %>% filter(year == yrs[i])

			if (time.unit == "weekly") {
				dat2 <- dat2 %>% mutate(y = if (y.unit == "count") pmin(call_count, cap) else call_count / sum(call_count, na.rm = TRUE),
					over = y.unit == "count" & call_count > cap)

				ggplot(dat2, aes(week_start, y, color = dx, group = dx)) +
					geom_line(linewidth = 0.9) +
					{ if (y.unit == "count") geom_text(data = filter(dat2, over), aes(label = "*"), vjust = -0.5, show.legend = FALSE) } +
					scale_color_manual(values = dxs.grp.color[dxs], breaks = dxs, name = NULL, drop = FALSE) +
					scale_x_date(breaks = date_breaks("3 months"), labels = date_format("%b", locale = "en")) +
					{ if (y.unit == "count") scale_y_continuous(limits = c(0, cap), breaks = seq(0, cap, cap/4))
					  else scale_y_continuous(labels = percent_format(accuracy = 1)) } +
					labs(title = yrs[i], x = NULL, y = if (i %% 2 == 1) ifelse(y.unit == "count", "Number of Calls", "Percentage") else NULL) +
					theme_minimal(base_size = 11) +
					theme(axis.title = element_text(face = "bold"), axis.text = element_text(face = "bold"), axis.line = element_line(), plot.title = element_text(face = "bold"))
			} else {
				ggplot(dat2, aes(hour, if (y.unit == "count") pmin(call_count, cap) else pct, color = dx, group = dx)) +
					geom_line(linewidth = 0.9) +
					geom_point(size = 1.8) +
					scale_color_manual(values = dxs.grp.color[dxs], breaks = dxs, name = NULL, drop = FALSE) +
					scale_x_continuous(breaks = seq(0, 22, 2), labels = sprintf("%02d", seq(0, 22, 2))) +
					{ if (y.unit == "count") scale_y_continuous(limits = c(0, cap), breaks = seq(0, cap, cap/4))
					  else scale_y_continuous(labels = percent_format(accuracy = 1)) } +
					labs(title = yrs[i], x = NULL, y = if (i %% 2 == 1) ifelse(y.unit == "count", "Number of Calls", "Percentage") else NULL) +
					theme_minimal(base_size = 11) +
					theme(axis.title = element_text(face = "bold"), axis.text = element_text(face = "bold"), axis.line = element_line(),
						axis.text.x = element_text(angle = 45, hjust = 1))
			}
		})
		p <- wrap_plots(plots, nrow = ceiling(length(yrs)/2), ncol = 2, guides = "collect") &
			theme(legend.position = "bottom", legend.text = element_text(face = "bold", size = 12)) &
			guides(color = guide_legend(nrow = 2, byrow = TRUE, override.aes = list(linewidth = 2)))
	} else {
		plots <- lapply(seq_along(dxs), \(i){
			dat2 <- dat1 %>% filter(dx == dxs[i])
			if (time.unit == "weekly") {
				dat2 <- dat2 %>% mutate(y = if (y.unit == "count") pmin(call_count, cap) else ave(call_count, year, FUN = \(x) x / sum(x)),
					over = y.unit == "count" & call_count > cap)

				ggplot(dat2, aes(week, y, color = factor(year), group = year)) +
					geom_line(linewidth = 0.9) +
					{ if (y.unit == "count") geom_text(data = filter(dat2, over), aes(label = "*"), vjust = -0.5, show.legend = FALSE) } +
					scale_color_manual(values = rainbow(length(yrs), s = 0.8, v = 0.85), breaks = yrs, name = NULL, drop = FALSE) +
					scale_x_continuous(breaks = c(1, 9, 18, 27, 36, 45), labels = c("Jan", "Mar", "May", "Jul", "Sep", "Nov")) +
					{ if (y.unit == "count") scale_y_continuous(limits = c(0, cap), breaks = seq(0, cap, cap/4))
					  else scale_y_continuous(labels = percent_format(accuracy = 1)) } +
					labs(title = dxs[i], x = NULL, y = if (i %% 2 == 1) ifelse(y.unit == "count", "Number of Calls", "Percentage") else NULL) +
					theme_minimal(base_size = 11) +
					theme(axis.title = element_text(face = "bold"), axis.text = element_text(face = "bold"), axis.line = element_line(), plot.title = element_text(face = "bold"))
			} else {
				ggplot(dat2, aes(hour, if (y.unit == "count") pmin(call_count, cap) else pct, color = factor(year), group = year)) +
					geom_line(linewidth = 0.9) +
					geom_point(size = 1.6) +
					scale_color_manual(values = rainbow(length(yrs), s = 0.8, v = 0.85), breaks = yrs, name = NULL, drop = FALSE) +
					scale_x_continuous(breaks = seq(0, 22, 2), labels = sprintf("%02d", seq(0, 22, 2))) +
					{ if (y.unit == "count") scale_y_continuous(limits = c(0, cap), breaks = seq(0, cap, cap/4))
					  else scale_y_continuous(labels = percent_format(accuracy = 1)) } +
					labs(title = dxs[i], x = NULL, y = if (i %% 2 == 1) ifelse(y.unit == "count", "Number of Calls", "Percentage") else NULL) +
					theme_minimal(base_size = 11) +
					theme(axis.title = element_text(face = "bold"), axis.text = element_text(face = "bold"), axis.line = element_line(),
						axis.text.x = element_text(angle = 45, hjust = 1))
			}
		})
		p <- wrap_plots(plots, nrow = ceiling(length(dxs)/2), ncol = 2, guides = "collect") &
			theme(legend.position = "bottom", legend.text = element_text(face = "bold", size = 12)) &
			guides(color = guide_legend(nrow = 2, byrow = TRUE, override.aes = list(linewidth = 2)))
	}
	if (!is.na(out_png)) ggsave(out_png, p, width = width, height = height, dpi = dpi)
	cat("\n===== Fig1 summary =====\n")
	if (time.unit == "weekly") {
		if (y.unit == "count") dat2 <- dat1 %>% group_by(dx) %>% summarise(total_calls = sum(call_count), mean_week = mean(call_count), sd_week = sd(call_count), weeks = n(), .groups = "drop")
		else dat2 <- dat1 %>% group_by(dx) %>% summarise(mean_pct = mean(if ("pct" %in% names(.)) pct else NA_real_, na.rm = TRUE), .groups = "drop")
	} else {
		if (y.unit == "count") dat2 <- dat1 %>% group_by(dx) %>% summarise(total_calls = sum(call_count), mean_hour = mean(call_count), sd_hour = sd(call_count), hours = n(), .groups = "drop")
		else dat2 <- dat1 %>% group_by(dx) %>% summarise(mean_pct = mean(pct, na.rm = TRUE), max_pct = max(pct, na.rm = TRUE), .groups = "drop")
	}
	print(as.data.frame(dat2))
	invisible(p)
}

Fig1 <- plot_dx_trend(dat1.list, yrs = 2017:2024, dx_var = "dx_grp", dxs = dxs.grp, time.unit = "weekly", var.group = "dxs", cap = 1000, out_png = "Fig1.png"); Fig1
#Fig1b <- plot_dx_trend(dat1.list, yrs = 2017:2024, dx_var = "dx_grp", dxs = dxs.grp, time.unit = "weekly", y.unit = "pct", var.group = "years", cap = 1000, out_png = "Fig1.weekly.by_dx.png"); Fig1b
#Fig1c <- plot_dx_trend(dat1.list, yrs = 2017:2024, dx_var = "dx_grp", dxs = dxs.grp, time.unit = "hourly", y.unit = "pct", var.group = "dxs", cap = 1000, out_png = "Fig1.hourly.by_year.png"); Fig1c
FigS1 <- plot_dx_trend(dat1.list, yrs = 2017:2024, dx_var = "dx_grp", dxs = dxs.grp, time.unit = "hourly", y.unit = "pct", var.group = "years", cap = 1000, out_png = "FigS1.png"); FigS1


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 🚩 图2. 🕒疾病类型每小时波动情况（circular）
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ems_hourly <- function(year = "2022", dxs = dxs.grp, dxs.color = dxs.grp.color, var.split = NA, var.split.keep = c("low", "high"),
	out_png = NA, width = 1200, height = 1200, res = 200, cex.lab = 0.7, sig = TRUE) {
	year <- as.character(year)
	dat1 <- bind_rows(lapply(year, \(y){
		d0 <- dat1.list[[y]]
		if (is.null(d0) || nrow(d0) == 0) return(tibble())
		d0 %>% filter(dx_grp %in% dxs, !is.na(hour)) %>%
			mutate(dx_grp = factor(as.character(dx_grp), levels = dxs), hour = as.integer(hour))
	}))
	if (nrow(dat1) == 0) stop("Fig2: No data after filtering.")
	if (all(is.na(var.split))) {
		dat2 <- dat1 %>%
			count(dx_grp, hour, name = "n") %>% complete(dx_grp, hour = 0:23, fill = list(n = 0)) %>%
			group_by(dx_grp) %>% mutate(N = sum(n), pct = (n + 0.5) / (N + 24 * 0.5)) %>% ungroup()
		bg_col <- setNames(adjustcolor(dxs.color[dxs], alpha.f = 0.18), dxs)
		ylim_top <- max(dat2$pct, na.rm = TRUE) * 1.06
		ylim_bot <- min(dat2$pct, na.rm = TRUE) * 0.94
	} else {
		if (!var.split %in% names(dat1)) stop("Fig2: var.split not found in dat1.")
		lev.ok <- sort(unique(na.omit(as.character(dat1[[var.split]]))))
		if (!all(var.split.keep %in% lev.ok)) stop(paste0("Fig2: var.split.keep must exist in ", var.split, ". Existing levels: ", paste(lev.ok, collapse = ", ")))
		if (length(var.split.keep) != 2) stop("Fig2: var.split.keep must have exactly 2 levels.")
		dat2 <- dat1 %>% filter(.data[[var.split]] %in% var.split.keep) %>%
			mutate(group = factor(as.character(.data[[var.split]]), levels = var.split.keep)) %>%
			count(dx_grp, group, hour, name = "n") %>% complete(dx_grp, group, hour = 0:23, fill = list(n = 0)) %>%
			group_by(dx_grp, group) %>% mutate(N = sum(n), pct = (n + 0.5) / (N + 24 * 0.5)) %>% ungroup() %>%
			select(dx_grp, group, hour, n, N, pct) %>% pivot_wider(names_from = group, values_from = c(n, N, pct)) %>%
			mutate(enrich = .data[[paste0("pct_", var.split.keep[2])]] / .data[[paste0("pct_", var.split.keep[1])]],
				p = pmap_dbl(list(.data[[paste0("n_", var.split.keep[2])]], .data[[paste0("N_", var.split.keep[2])]], .data[[paste0("n_", var.split.keep[1])]], .data[[paste0("N_", var.split.keep[1])]]),
					\(a, A, b, B) if (A == 0 || B == 0) NA_real_ else fisher.test(matrix(c(a, A-a, b, B-b), 2, byrow = TRUE))$p.value)) %>%
			group_by(dx_grp) %>% mutate(p_adj = p.adjust(p, "BH"), sig = !is.na(p_adj) & p_adj < 0.05) %>% ungroup()
		bg_col <- setNames(adjustcolor(dxs.color[dxs], alpha.f = 0.18), dxs)
		ylim_top <- max(1.25, max(dat2$enrich, na.rm = TRUE))
		ylim_bot <- min(0.80, 1 / ylim_top)
	}
	if (!is.na(out_png)) png(out_png, width = width, height = height, res = res)
	circos.clear()
	circos.par(start.degree = 90, gap.degree = 2, cell.padding = c(0,0,0,0), track.margin = c(0.002, 0.002))
	circos.initialize(factors = "all", xlim = c(0, 24))
	for (i in seq_along(dxs)) {
		dx <- dxs[i]
		dt <- dat2 %>% filter(dx_grp == dx) %>% arrange(hour)
		circos.trackPlotRegion(
			factors = "all", track.index = i, ylim = c(ylim_bot, ylim_top),
			bg.col = bg_col[dx], bg.border = NA, track.height = min(0.16, 0.92 / length(dxs)),
			panel.fun = function(...) {
				if (all(is.na(var.split))) {
					circos.lines(dt$hour, dt$pct, col = dxs.color[dx], lwd = 2.3)
				} else {
					circos.lines(c(0, 23), c(1, 1), col = "grey60", lty = 3, lwd = 2)
					circos.lines(dt$hour, dt$enrich, col = dxs.color[dx], lwd = 2.3)
					if (sig && any(dt$sig, na.rm = TRUE)) {
						dd <- dt[dt$sig, ]
						yy <- pmin(dd$enrich * 1.04, ylim_top * 0.98)
						circos.text(dd$hour, yy, "*", cex = 0.45, font = 2, col = dxs.color[dx])
					}
				}
				circos.text(23.55, ylim_top * 0.85, dx, facing = "bending.inside", niceFacing = TRUE, adj = c(0.5, 0.5), cex = cex.lab, font = 2, col = dxs.color[dx])
			}
		)
	}
	circos.axis(h = "top", major.at = 0:23, labels = sprintf("%02d", 0:23), labels.cex = 1.1, minor.ticks = 0, sector.index = "all", track.index = 1)
	if (!is.na(out_png)) dev.off()
	cat("\n===== Fig2 summary =====\n")
	if (all(is.na(var.split))) {
		dat3 <- dat2 %>% group_by(dx_grp) %>% slice_max(order_by = pct, n = 3, with_ties = FALSE) %>% arrange(dx_grp, desc(pct))
		print(as.data.frame(dat3))
	} else {
		dat3 <- dat2 %>% group_by(dx_grp) %>% summarise(
			enrich_mean = mean(enrich, na.rm = TRUE),
			enrich_min = min(enrich, na.rm = TRUE),
			enrich_max = max(enrich, na.rm = TRUE),
			n_sig_hr = sum(sig, na.rm = TRUE),
			.groups = "drop"
		)
		print(as.data.frame(dat3))
	}
	print(recordPlot())
	invisible(as.data.frame(dat2))
}
dxs1 <- c("CVD", "Respiratory", "Mental", "Poison", "Death")
Fig2 <- ems_hourly(year = "2020", dxs = dxs1, dxs.color = dxs.grp.color[dxs1], out_png = "Fig2.png"); Fig2
# Fig2b <- ems_hourly(year = "2024", dxs = dxs1, dxs.color = dxs.grp.color[dxs1], var.split = "phone.luck", var.split.keep = c("low","high"))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 🚩 图3. 🎭📱两组人的12年发病频率
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cap <- 1.2; xN <- 1.15
s2 <- bind_rows(lapply(years, \(y) {
	d0 <- dat1.list[[as.character(y)]]; if (is.null(d0) || nrow(d0) == 0) return(tibble())
	d0 <- d0 %>% filter(phone.luck %in% grp_use, !is.na(疾病分类.ML)) %>%
		transmute(group = as.character(phone.luck), dx_raw = trimws(疾病分类.ML)) %>% left_join(map_grp, by = "dx_raw") %>% filter(dx_grp %in% dxs.grp)
	tg <- table(d0$dx_grp, d0$group); ta <- table(d0$dx_grp); Nl <- sum(tg[, "low"]); Nh <- sum(tg[, "high"]); Tall <- sum(ta)
	tibble(year = y, disease = dxs.grp,
		n_low = as.numeric(tg[dxs.grp, "low"]), n_high = as.numeric(tg[dxs.grp, "high"]),
		N_low = Nl, N_high = Nh,
		pct_all = as.numeric(ta[dxs.grp]) / Tall, pct_low = as.numeric(tg[dxs.grp, "low"]) / Nl, pct_high = as.numeric(tg[dxs.grp, "high"]) / Nh
	) %>% mutate(
		enrich_low = pct_low / pct_all, enrich_high = pct_high / pct_all, RR = pct_high / pct_low,
		p = purrr::pmap_dbl(list(n_high, N_high, n_low, N_low), \(a, A, b, B) suppressWarnings(chisq.test(matrix(c(a, A - a, b, B - b), 2))$p.value)),
		sig = !is.na(p) & p < 0.01, lo = pmax(enrich_low,  1 / cap), hi = pmin(enrich_high, cap), lf = enrich_low  < 1 / cap, hf = enrich_high > cap
	)
}))
print(as.data.frame(s2)) # 🏮

plots <- lapply(dxs.grp, \(dx) {
	col <- dxs.grp.color[dx]; d <- s2 %>% filter(disease == dx)
	ggplot(d, aes(y = year)) + geom_vline(xintercept = 1) +
		geom_vline(xintercept = xN, linetype = "dashed", color = "grey40", linewidth = 0.6) +
		geom_segment(aes(x = 1, xend = lo, yend = year), linetype = "dashed", color = "grey70", linewidth = 0.9) +
		geom_segment(aes(x = 1, xend = hi, yend = year), linetype = "dashed", color = col, linewidth = 0.9) +
		geom_point(aes(x = lo), color = "grey50", size = 3) + geom_point(aes(x = hi), color = col, size = 3) +
		geom_text(data = d %>% filter(sig), aes(x = hi, label = "*"), hjust = -0.2, vjust = 0.3, size = 5, fontface = "bold") +
		geom_text(data = d %>% filter(lf),  aes(x = lo, label = "<"), hjust = 1.2) +
		geom_text(data = d %>% filter(hf),  aes(x = hi, label = ">"), hjust = 0) +
	geom_text(aes(x = xN - 0.005, label = n_low), hjust = 1, size = 3.2, fontface = "bold", color = "grey60") +
	geom_text(aes(x = xN, label = "    "), hjust = 0.5, size = 3.2, fontface = "bold", color = "black") +
	geom_text(aes(x = xN + 0.005, label = n_high), hjust = 0, size = 3.2, fontface = "bold", color = col)  +  
	scale_x_continuous(limits = c(0.9, 1.2)) + scale_y_continuous(breaks = years, labels = years) +
		labs(title = dx, x = NULL, y = NULL) + theme_minimal() +
		theme(axis.text = element_text(face = "bold"), axis.title = element_text(face = "bold"), axis.line = element_line(), legend.position = "none")
})
Fig3 <- wrap_plots(plots, nrow = 4, ncol = 2)
Fig3; ggsave("Fig3.png", Fig3, width = 11.2, height = 10, dpi = 600)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 🚩 图4. 急救🚑时间 (Dispatch, Drive, Onsite)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ems_duration <- function(var.time = "Dispatch", years = years, dxs = dxs.grp, var.group = "dxs",
	var.split = NA, var.split.keep = c("low", "high"), out_png = NA, width = 6, height = 8, dpi = 600) {
	v.map <- c("Dispatch" = "派车时间", "Driving" = "去程时间", "Onsite" = "现场时间")
	v0 <- v.map[var.time]
	dat1 <- bind_rows(lapply(years, \(y){
		d0 <- dat1.list[[as.character(y)]]
		if (is.null(d0) || nrow(d0) == 0) return(tibble())
		d0 %>% filter(dx_grp %in% dxs) %>%
			transmute(
				Year = y,
				dx_grp = factor(as.character(dx_grp), levels = dxs),
				phone.luck = if ("phone.luck" %in% names(.)) as.character(phone.luck) else NA_character_,
				X = suppressWarnings(as.numeric(.data[[v0]]) / 60)
			)
	}))
	if (nrow(dat1) == 0) stop("Fig3: No data after filtering.")
	if (all(is.na(var.split))) {
		if (var.group == "dxs") {
			dat2 <- dat1 %>% group_by(Year, dx_grp) %>% summarise(mean = mean(X, na.rm = TRUE), .groups = "drop")
			p <- ggplot(dat2, aes(x = mean, y = Year, color = dx_grp)) +
				geom_point(size = 3) +
				scale_color_manual(values = dxs.grp.color[dxs], name = NULL, drop = FALSE) +
				scale_y_continuous(breaks = years, labels = years) +
				labs(title = var.time, x = "Time (mins)", y = "Year") +
				theme_minimal(base_size = 12) +
				theme(axis.title = element_text(face = "bold"), axis.text = element_text(face = "bold"), axis.line = element_line(), plot.title = element_text(face = "bold"))
		} else {
			dat2 <- dat1 %>% group_by(dx_grp, Year) %>% summarise(mean = mean(X, na.rm = TRUE), .groups = "drop")
			p <- ggplot(dat2, aes(x = mean, y = fct_rev(dx_grp), color = factor(Year))) +
				geom_point(size = 3) +
				scale_color_manual(values = rainbow(length(unique(dat2$Year)), s = 0.8, v = 0.85), name = NULL) +
				labs(title = var.time, x = "Time (mins)", y = NULL) +
				theme_minimal(base_size = 12) +
				theme(axis.title = element_text(face = "bold"), axis.text = element_text(face = "bold"), axis.line = element_line(), plot.title = element_text(face = "bold"))
		}
		cat("\n===== Fig3 summary =====\n")
		print(as.data.frame(dat2))
	} else {
		if (!var.split %in% names(dat1)) stop("Fig3: var.split not found in dat1.")
		lev.ok <- sort(unique(na.omit(as.character(dat1[[var.split]]))))
		if (!all(var.split.keep %in% lev.ok)) stop(paste0("Fig3: var.split.keep must exist in ", var.split, ". Existing levels: ", paste(lev.ok, collapse = ", ")))
		if (length(var.split.keep) != 2) stop("Fig3: var.split.keep must have exactly 2 levels.")
		if (var.group == "dxs") {
			dat2 <- dat1 %>% filter(.data[[var.split]] %in% var.split.keep) %>%
				group_by(Year, split = .data[[var.split]]) %>%
				summarise(mean = mean(X, na.rm = TRUE), .groups = "drop") %>%
				pivot_wider(names_from = split, values_from = mean)
			names(dat2)[names(dat2) == var.split.keep[1]] <- "low"
			names(dat2)[names(dat2) == var.split.keep[2]] <- "high"
			dat2 <- dat2 %>% mutate(
				diff = high - low,
				p = pmap_dbl(list(Year), \(yy){
					d0 <- dat1 %>% filter(Year == yy, .data[[var.split]] %in% var.split.keep)
					vl <- d0$X[d0[[var.split]] == var.split.keep[1]]
					vh <- d0$X[d0[[var.split]] == var.split.keep[2]]
					suppressWarnings(tryCatch(wilcox.test(vh, vl)$p.value, error = \(e) NA_real_))
				})
			) %>% mutate(p_adj = p.adjust(p, "BH"), sig = !is.na(p_adj) & p_adj < 0.01)
			gm <- mean(dat2$high, na.rm = TRUE); rng <- range(c(dat2$low, dat2$high), na.rm = TRUE); eps <- 0.05 * diff(rng)
			dd <- dat2 %>% mutate(star_x = ifelse(high >= low, high + eps, low - eps))
			p <- ggplot(dd, aes(y = Year)) +
				geom_segment(aes(x = low, xend = high, yend = Year), color = "black", linewidth = 0.5) +
				geom_point(aes(x = low),  color = "grey50", size = 3) +
				geom_point(aes(x = high), color = "blue", size = 3, shape = 17) +
				geom_vline(xintercept = gm, color = "blue", linetype = "dashed", linewidth = 0.9) +
				geom_text(data = dplyr::filter(dd, sig), aes(x = star_x, label = "*"), fontface = "bold", size = 5, vjust = 0.35) +
				labs(title = var.time, x = "Time (mins)", y = "Year") +
				scale_y_continuous(breaks = years, labels = years) +
				theme_minimal(base_size = 12) +
				theme(axis.title = element_text(face = "bold"), axis.text = element_text(face = "bold"), axis.line = element_line(), plot.title = element_text(face = "bold"))
		} else {
			dat2 <- dat1 %>% filter(.data[[var.split]] %in% var.split.keep, dx_grp %in% dxs) %>%
				group_by(dx_grp, split = .data[[var.split]]) %>%
				summarise(mean = mean(X, na.rm = TRUE), .groups = "drop") %>%
				pivot_wider(names_from = split, values_from = mean)
			names(dat2)[names(dat2) == var.split.keep[1]] <- "low"
			names(dat2)[names(dat2) == var.split.keep[2]] <- "high"
			dat2 <- dat2 %>% mutate(
				diff = high - low,
				p = pmap_dbl(list(dx_grp), \(ddx){
					d0 <- dat1 %>% filter(dx_grp == ddx, .data[[var.split]] %in% var.split.keep)
					vl <- d0$X[d0[[var.split]] == var.split.keep[1]]
					vh <- d0$X[d0[[var.split]] == var.split.keep[2]]
					suppressWarnings(tryCatch(wilcox.test(vh, vl)$p.value, error = \(e) NA_real_))
				})
			) %>% mutate(p_adj = p.adjust(p, "BH"), sig = !is.na(p_adj) & p_adj < 0.01)
			gm <- mean(dat2$high, na.rm = TRUE); rng <- range(c(dat2$low, dat2$high), na.rm = TRUE); eps <- 0.05 * diff(rng)
			dd <- dat2 %>% mutate(star_x = ifelse(high >= low, high + eps, low - eps))
			p <- ggplot(dd, aes(y = fct_rev(dx_grp))) +
				geom_segment(aes(x = low, xend = high, yend = fct_rev(dx_grp)), color = "black", linewidth = 0.5) +
				geom_point(aes(x = low),  color = "grey50", size = 3) +
				geom_point(aes(x = high), color = "blue", size = 3, shape = 17) +
				geom_vline(xintercept = gm, color = "blue", linetype = "dashed", linewidth = 0.9) +
				geom_text(data = dplyr::filter(dd, sig), aes(x = star_x, label = "*"), fontface = "bold", size = 5, vjust = 0.35) +
				labs(title = var.time, x = "Time (mins)", y = NULL) +
				theme_minimal(base_size = 12) +
				theme(axis.title = element_text(face = "bold"), axis.text = element_text(face = "bold"), axis.line = element_line(), plot.title = element_text(face = "bold"))
		}
		cat("\n===== Fig3 summary =====\n")
		print(as.data.frame(dat2))
	}
	if (!is.na(out_png)) ggsave(out_png, p, width = width, height = height, dpi = dpi)
	p
}

dxs1 <- dxs.grp # c("CVD", "Respiratory", "Mental", "Death")
Fig4a <- ems_duration(var.time = "Onsite", years = years, dxs = dxs1, var.group = "dxs", var.split = "phone.luck", var.split.keep = c("low","high"), out_png = "Fig4a.png"); Fig4a
Fig4b <- ems_duration(var.time = "Onsite", years = years, dxs = dxs1, var.group = "dxs", out_png = "Fig4b.png"); Fig4b


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 🚩 图5A. 疫情管控影响 🚫
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
phsm_in <- function(date_begin = as.Date("2022-03-14"), date_end = as.Date("2022-03-20"), flank_days = 7,
	dxs = dxs.grp, var.split = NA, var.split.keep = c("low", "high"),
	out_png = NA, width = 10, height = 10, dpi = 600) {
	dat1 <- dat1.list[[as.character(year(date_begin))]] %>%
		transmute(
			日期 = as.Date(日期),
			dx_grp = factor(as.character(dx_grp), levels = dxs),
			split = if (all(is.na(var.split))) "All" else as.character(.data[[var.split]])
		) %>%
		filter(dx_grp %in% dxs, between(日期, date_begin - flank_days, date_end + flank_days))
	if (!all(is.na(var.split))) dat1 <- dat1 %>% filter(split %in% var.split.keep)
	dat2 <- dat1 %>% count(dx_grp, split, 日期, name = "count") %>%
		group_by(dx_grp, split) %>% arrange(日期) %>% mutate(count3 = roll3(count)) %>% ungroup() %>%
		mutate(
			period = case_when(日期 < date_begin ~ "pre", between(日期, date_begin, date_end) ~ "during", 日期 > date_end ~ "after"),
			period = factor(period, c("pre", "during", "after")),
			col = if (all(is.na(var.split))) as.character(dx_grp) else ifelse(split == var.split.keep[1], "low", as.character(dx_grp))
		)
	FigA <- ggplot(dat2, aes(日期, count3, color = col, group = interaction(dx_grp, split))) +
		geom_vline(xintercept = c(date_begin, date_end), linetype = "dashed", color = "orange", linewidth = 1) +
		geom_line(linewidth = 1, na.rm = TRUE) +
		scale_color_manual(values = c("low" = "grey60", dxs.grp.color[dxs]), drop = FALSE) +
		facet_wrap(~dx_grp, scales = "free_y", ncol = 1) +
		scale_x_date(labels = date_format("%b %d", locale = "en")) +
		scale_y_continuous(breaks = pretty_breaks(n = 2)) +
		labs(title = "A. Daily Calls (3-day avg)", x = NULL, y = NULL) +
		theme_minimal(base_size = 12) +
		theme(axis.text = element_text(face = "bold"), axis.title = element_text(face = "bold"), strip.text = element_text(face = "bold"), legend.position = "none", plot.title = element_text(face = "bold"))
	fit_period4 <- function(df){
		fit <- glm(count ~ period, family = poisson, data = df)
		td <- broom::tidy(fit)
		pick <- function(term_use){
			x <- td %>% filter(term == term_use)
			if (nrow(x) == 0) return(tibble(term = term_use, RR = NA_real_, lo = NA_real_, hi = NA_real_, p = NA_real_))
			tibble(term = term_use, RR = exp(x$estimate), lo = exp(x$estimate - 1.96 * x$std.error), hi = exp(x$estimate + 1.96 * x$std.error), p = x$p.value)
		}
		bind_rows(pick("periodduring"), pick("periodafter"))
	}
	dat3 <- dat2 %>% group_by(dx_grp, split) %>% nest() %>% mutate(res = map(data, fit_period4)) %>% unnest(res) %>%
		mutate(
			phase = if_else(str_detect(term, "periodduring"), "During PHSM", "After PHSM"),
			sig = sig_star(p),
			dx_grp = fct_relevel(dx_grp, dxs),
			col = if (all(is.na(var.split))) as.character(dx_grp) else ifelse(split == var.split.keep[1], "low", as.character(dx_grp))
		)
	FigB <- function(tt, d){
		ggplot(d, aes(x = RR, y = fct_rev(dx_grp), color = col)) +
			geom_vline(xintercept = 1, linetype = "dashed") +
			geom_point(size = 3, position = position_dodge(width = 0.4)) +
			geom_errorbar(aes(xmin = lo, xmax = hi), width = 0.2, orientation = "y", position = position_dodge(width = 0.4)) +
			geom_text(aes(label = sig), hjust = -0.4, position = position_dodge(width = 0.4)) +
			scale_color_manual(values = c("low" = "grey60", dxs.grp.color[dxs]), drop = FALSE) +
			labs(title = tt, x = expression(italic("Rate ratio (vs pre)")), y = NULL) +
			theme_minimal(base_size = 12) +
			theme(axis.text = element_text(face = "bold"), axis.title = element_text(face = "bold"), strip.text = element_text(face = "bold"), legend.position = "none", plot.title = element_text(face = "bold"))
	}
	FigB1 <- FigB("B. During PHSM", filter(dat3, phase == "During PHSM"))
	FigB2 <- FigB("C. After PHSM", filter(dat3, phase == "After PHSM"))
	Fig <- (FigA | (FigB1 / FigB2)) + plot_layout(widths = c(2, 1))
	if (!is.na(out_png)) ggsave(out_png, Fig, width = width, height = height, dpi = dpi)
	cat("\n===== Fig5A summary =====\n")
	print(as.data.frame(dat3 %>% select(dx_grp, split, phase, RR, lo, hi, p, sig) %>% arrange(phase, dx_grp, split)))
	Fig
}

Fig5a <- phsm_in(date_begin = as.Date("2022-03-14"), date_end = as.Date("2022-03-20"), flank_days = 7, var.split = "phone.luck", var.split.keep = c("low","high"), out_png = "Fig5a.png"); Fig5a


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 🚩 图5B. 疫情放开影响🎇
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
phsm_out <- function(date_half = as.Date("2022-11-11"), date_full = as.Date("2022-12-07"), flank_days = 10,
	dxs = dxs.grp, var.split = NA, var.split.keep = c("low", "high"),
	out_png = NA, width = 10, height = 10, dpi = 600) {
	dat1 <- bind_rows(
		dat1.list[[as.character(year(date_half))]] %>%
			transmute(
				日期 = as.Date(日期),
				dx_grp = factor(as.character(dx_grp), levels = dxs),
				split = if (all(is.na(var.split))) "All" else as.character(.data[[var.split]])
			) %>%
			filter(dx_grp %in% dxs, between(日期, date_half - flank_days, as.Date("2022-12-31"))),
		dat1.list[[as.character(year(date_half) + 1)]] %>%
			transmute(
				日期 = as.Date(日期),
				dx_grp = factor(as.character(dx_grp), levels = dxs),
				split = if (all(is.na(var.split))) "All" else as.character(.data[[var.split]])
			) %>%
			filter(dx_grp %in% dxs, between(日期, as.Date("2023-01-01"), date_full + 34))
	)
	if (!all(is.na(var.split))) dat1 <- dat1 %>% filter(split %in% var.split.keep)
	dat2 <- dat1 %>% count(dx_grp, split, 日期, name = "count") %>%
		group_by(dx_grp, split) %>% arrange(日期) %>% mutate(count3 = roll3(count)) %>% ungroup() %>%
		mutate(
			period = case_when(日期 < date_half ~ "pre", 日期 >= date_half & 日期 < date_full ~ "mid", 日期 >= date_full & 日期 <= date_full + 24 ~ "post", TRUE ~ NA_character_),
			period = factor(period, c("pre", "mid", "post")),
			col = if (all(is.na(var.split))) as.character(dx_grp) else ifelse(split == var.split.keep[1], "low", as.character(dx_grp))
		) %>% filter(!is.na(period))
	FigA <- ggplot(dat2, aes(日期, count3, color = col, group = interaction(dx_grp, split))) +
		geom_vline(xintercept = c(date_half, date_full), linetype = "dashed", color = "orange", linewidth = 1) +
		geom_line(linewidth = 1, na.rm = TRUE) +
		scale_color_manual(values = c("low" = "grey60", dxs.grp.color[dxs]), drop = FALSE) +
		facet_wrap(~dx_grp, scales = "free_y", ncol = 1) +
		scale_x_date(labels = date_format("%b %d", locale = "en")) +
		scale_y_continuous(breaks = pretty_breaks(n = 2)) +
		labs(title = "A. Daily Calls (3-day avg)", x = NULL, y = NULL) +
		theme_minimal(base_size = 12) +
		theme(axis.text = element_text(face = "bold"), axis.title = element_text(face = "bold"), strip.text = element_text(face = "bold"), legend.position = "none", plot.title = element_text(face = "bold"))
	fit_period5 <- function(df){
		fit <- glm(count ~ period, family = poisson, data = df)
		td <- broom::tidy(fit)
		pick <- function(term_use){
			x <- td %>% filter(term == term_use)
			if (nrow(x) == 0) return(tibble(term = term_use, RR = NA_real_, lo = NA_real_, hi = NA_real_, p = NA_real_))
			tibble(term = term_use, RR = exp(x$estimate), lo = exp(x$estimate - 1.96 * x$std.error), hi = exp(x$estimate + 1.96 * x$std.error), p = x$p.value)
		}
		bind_rows(pick("periodmid"), pick("periodpost"))
	}
	dat3 <- dat2 %>% group_by(dx_grp, split) %>% nest() %>% mutate(res = map(data, fit_period5)) %>% unnest(res) %>%
		mutate(
			phase = if_else(str_detect(term, "periodmid"), "First open-up (mid)", "Final open-up (post)"),
			sig = sig_star(p),
			dx_grp = fct_relevel(dx_grp, dxs),
			col = if (all(is.na(var.split))) as.character(dx_grp) else ifelse(split == var.split.keep[1], "low", as.character(dx_grp))
		)
	FigB <- function(tt, d){
		ggplot(d, aes(x = RR, y = fct_rev(dx_grp), color = col)) +
			geom_vline(xintercept = 1, linetype = "dashed") +
			geom_point(size = 3, position = position_dodge(width = 0.4)) +
			geom_errorbar(aes(xmin = lo, xmax = hi), width = 0.2, orientation = "y", position = position_dodge(width = 0.4)) +
			geom_text(aes(label = sig), hjust = -0.4, position = position_dodge(width = 0.4)) +
			scale_color_manual(values = c("low" = "grey60", dxs.grp.color[dxs]), drop = FALSE) +
			labs(title = tt, x = expression(italic("Rate ratio (vs pre)")), y = NULL) +
			theme_minimal(base_size = 12) +
			theme(axis.text = element_text(face = "bold"), axis.title = element_text(face = "bold"), strip.text = element_text(face = "bold"), legend.position = "none", plot.title = element_text(face = "bold"))
	}
	FigB1 <- FigB("B. First open-up (mid)", filter(dat3, phase == "First open-up (mid)"))
	FigB2 <- FigB("C. Final open-up (post)", filter(dat3, phase == "Final open-up (post)"))
	Fig <- (FigA | (FigB1 / FigB2)) + plot_layout(widths = c(2, 1))
	if (!is.na(out_png)) ggsave(out_png, Fig, width = width, height = height, dpi = dpi)
	cat("\n===== Fig5B summary =====\n")
	print(as.data.frame(dat3 %>% select(dx_grp, split, phase, RR, lo, hi, p, sig) %>% arrange(phase, dx_grp, split)))
	Fig
}

Fig5b <- phsm_out(date_half = as.Date("2022-11-11"), date_full = as.Date("2022-12-07"), var.split = "phone.luck", var.split.keep = c("low","high")); Fig5b


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 🚩 图6. 幸运者的房价🏠
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pacman::p_load(sf)
year_use <- "2021"; dist_max_m <- 1000
house_sf <- read_excel(file.path(dir.dat, "深圳房价.xlsx")) %>%
	transmute(house.id = row_number(), house.price = as.numeric(房价), Lon = as.numeric(Lon), Lat = as.numeric(Lat)) %>%
	filter(is.finite(Lon), is.finite(Lat), is.finite(house.price)) %>%
	st_as_sf(coords = c("Lon", "Lat"), crs = 4326, remove = FALSE) %>% st_transform(3857)
X_sf <- dat1.list[[year_use]] %>%
	transmute(X.id = row_number(), 地址类型, phone.sco = as.numeric(phone.sco), phone.luck = as.character(phone.luck), lon = as.numeric(接车地址经度), lat = as.numeric(接车地址纬度)) %>%
	filter(地址类型 == "住宅区", phone.luck %in% c("low", "high"), is.finite(lon), is.finite(lat)) %>%
	st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE) %>% st_transform(3857)

idx <- st_nearest_feature(X_sf, house_sf)
dat0 <- X_sf %>% mutate(house.price = house_sf$house.price[idx], dist_m = as.numeric(st_distance(X_sf, house_sf[idx, ], by_element = TRUE))) %>%
	st_drop_geometry() %>% mutate(house.price = if_else(dist_m <= dist_max_m, house.price, NA_real_))
print(as.data.frame(dat0 %>% summarise(
	N = n(), N_price_ok = sum(is.finite(house.price)), pct_ok = mean(is.finite(house.price)),
	dist_med = median(dist_m, na.rm = TRUE), dist_p90 = as.numeric(quantile(dist_m, 0.9, na.rm = TRUE))
)))
dat_bin <- dat0 %>% filter(is.finite(house.price), is.finite(phone.sco)) %>% mutate(logp = log10(house.price))
h <- hist(dat_bin$logp, breaks = 10, plot = FALSE)
s6 <- dat_bin %>% mutate(bin = cut(logp, breaks = h$breaks, include.lowest = TRUE)) %>%
	group_by(bin) %>% summarise(x  = round(mean(range(logp)), 2), n = n(), y = round(mean(phone.sco), 2), sd = round(sd(phone.sco), 2), .groups = "drop") %>% arrange(x)
print(as.data.frame(s6)) # 🏮

s6_use <- s6 
png("Fig6.png", width = 10, height = 5, units = "in", res = 300)
par(mar = c(5, 4, 3, 5) + 0.2, font.lab = 2, font.axis = 2)
hh <- hist(dat_bin$logp, breaks = h$breaks, freq = TRUE, col = "grey85", border = "grey40", main = "", xlab = "log10(house price)", ylab = "")
par(new = TRUE)
plot(s6_use$x, s6_use$y, ylim = range(c(s6_use$y - s6_use$sd, s6_use$y + s6_use$sd), na.rm = TRUE),
		 xlim = range(hh$breaks), axes = FALSE, xlab = NA, ylab = NA, pch = 16, cex = 1.2, col = "blue")
arrows(s6_use$x, s6_use$y - s6_use$sd, s6_use$x, s6_use$y + s6_use$sd, angle = 90, code = 3, length = 0.05, col = "grey60")
axis(side = 4, font.axis = 2)
mtext(side = 4, line = 3, "Phone luck score", col = "blue", font = 2)
dev.off()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 🚩 图S2. 原始分类【中文】🎇
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plot_dx <- function(dat_list, years, dx_var, dxs.cn,
	level=c("raw","group"), show=c("percent","count")){
	level <- match.arg(level); show <- match.arg(show)
	dxs_raw <- unlist(dxs.cn, use.names=FALSE)
	map_grp <- stack(dxs.cn); colnames(map_grp) <- c("dx_raw","group")
	dat <- purrr::map_dfr(years, \(y){
		d <- dat_list[[as.character(y)]]
		if(is.null(d)||nrow(d)==0) return(tibble())
		d0 <- d %>% filter(!is.na(.data[[dx_var]])) %>% transmute(dx_raw=.data[[dx_var]])
		out <- if(level=="raw")
			d0 %>% filter(dx_raw%in%dxs_raw) %>% count(group=dx_raw,name="count")
		else
			d0 %>% left_join(map_grp,by="dx_raw") %>% filter(!is.na(group)) %>% count(group,name="count")
		out %>% mutate(year=y,pct=count/sum(count))
	})
	lev <- dat %>% filter(year==max(years)) %>% arrange(desc(pct)) %>% pull(group) %>% unique()
	dat <- dat %>% mutate(group=factor(group,levels=lev), y=if(show=="percent") pct else count)

	ggplot(dat,aes(year,y,color=group,group=group))+
		geom_line(linewidth=1)+geom_point(size=2)+
		scale_x_continuous(breaks=years)+
		(if(show=="percent")
			scale_y_continuous(labels=scales::percent_format(accuracy=1),expand=c(0,0))
		 else scale_y_continuous(expand=c(0,0)))+
		labs(x="Year",y=if(show=="percent")"Percentage" else "Count",color=NULL)+
		theme_minimal(base_size=12)
}

FigS2 <- plot_dx(dat1.list, years, "疾病分类.ML", dxs, level="raw",   show="percent") # count
FigS2; ggsave("FigS2.png", FigS2, width=7, height=4.2, dpi=300)
