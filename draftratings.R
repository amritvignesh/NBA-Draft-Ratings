library(cbbdata)
library(dplyr)
library(tidyverse)
library(xgboost)
library(caret)
library(vip)
library(cbbplotR)
library(gt)
library(gtExtras)

stats <- cbd_torvik_player_season() %>% filter(min >= 40)

stats_2024 <- stats %>% filter(year == 2024) 
players_2024 <- unique(stats_2024$player)

condensed_stats <- stats %>% filter(!is.na(pick) | (player %in% players_2024)) %>% select(year, player, team, pick)

last_years <- condensed_stats %>%
  group_by(player) %>%
  arrange(year) %>%
  summarize(last_year = last(year))

condensed_stats <- left_join(condensed_stats, last_years) %>%
  filter(last_year >= 2011, year >= last_year - 3)

drafted <- condensed_stats %>% filter(!is.na(pick))

# https://www.hoopsrumors.com/2024/03/2024-nba-draft-early-entrants-list.html
# players left off are ones playing for non-d1 teams, didnt meet the 40% min requirement, and deshawndre washington who didnt play the 23-24 season

draft_players <- read_csv("draft_players.csv")

draft_2024 <- condensed_stats %>% filter(is.na(pick), player %in% draft_players$player)

draft_2024 <- draft_2024 %>% filter(!(player == "Devin Carter" & team == "Alcorn St." | player == "David Jones" & team == "Sacramento St." | player == "David Jones" & team == "Georgia Southern"))

condensed_stats <- rbind(drafted, draft_2024) 

condensed_stats <- condensed_stats %>% group_by(player, pick) %>% mutate(season_recent_num = rank(-year))

final_stats <- stats %>% inner_join(condensed_stats, by = c("year", "player", "team")) %>% select(year, player, team, min, season_recent_num, porpag, dporpag, pick = pick.x) %>% select(-team, -min, -year)

final_stats_wide <- final_stats %>%
  pivot_wider(
    names_from = season_recent_num,
    values_from = c(porpag, dporpag),
    names_sep = "_"
  )

one_year <- final_stats_wide %>% filter(is.na(porpag_2)) %>% select_if(~any(!is.na(.))) 
two_year <- final_stats_wide %>% filter(is.na(porpag_3) & !is.na(porpag_2)) %>% select_if(~any(!is.na(.)))
three_year <- final_stats_wide %>% filter(is.na(porpag_4) & !is.na(porpag_3) & !is.na(porpag_2)) %>% select_if(~any(!is.na(.)))
four_year <- final_stats_wide %>% filter(!is.na(porpag_4)) %>% select_if(~any(!is.na(.)))

one_year_train <- one_year %>% filter(!is.na(pick))
two_year_train <- two_year %>% filter(!is.na(pick))
three_year_train <- three_year %>% filter(!is.na(pick))
four_year_train <- four_year %>% filter(!is.na(pick))

labels_1 <- as.matrix(one_year_train[, 2])
train_1 <- as.matrix(one_year_train[, c(3:4)])

model_1 <- xgboost(data = train_1, label = labels_1, nrounds = 100, objective = "reg:squarederror", early_stopping_rounds = 10, max_depth = 6, eta = 0.3)

labels_2 <- as.matrix(two_year_train[, 2])
train_2 <- as.matrix(two_year_train[, c(3:6)])

model_2 <- xgboost(data = train_2, label = labels_2, nrounds = 100, objective = "reg:squarederror", early_stopping_rounds = 10, max_depth = 6, eta = 0.3)

labels_3 <- as.matrix(three_year_train[, 2])
train_3 <- as.matrix(three_year_train[, c(3:8)])

model_3 <- xgboost(data = train_3, label = labels_3, nrounds = 100, objective = "reg:squarederror", early_stopping_rounds = 10, max_depth = 6, eta = 0.3)

labels_4 <- as.matrix(four_year_train[, 2])
train_4 <- as.matrix(four_year_train[, c(3:10)])

model_4 <- xgboost(data = train_4, label = labels_4, nrounds = 100, objective = "reg:squarederror", early_stopping_rounds = 10, max_depth = 6, eta = 0.3)

model_2_weights <- as.data.frame(vi(model_2))
model_3_weights <- as.data.frame(vi(model_3))
model_4_weights <- as.data.frame(vi(model_4))

weight_func <- function(df) {
  df <- df %>%
    mutate(stat = sub("(_\\d+)$", "", Variable)) %>%
    group_by(stat) %>%
    mutate(Importance = Importance / sum(Importance)) %>%
    ungroup() %>%
    select(-stat)
  return(df)
} 

model_2_weights <- weight_func(model_2_weights)
model_3_weights <- weight_func(model_3_weights)
model_4_weights <- weight_func(model_4_weights)

one_year_test <- one_year %>% filter(is.na(pick)) %>%
  mutate(off_score = porpag_1, def_score = dporpag_1) %>%
  select(-pick, -porpag_1, -dporpag_1)

two_year_test <- two_year %>% filter(is.na(pick)) %>%
  mutate(off_score = porpag_1 * model_2_weights$Importance[which(model_2_weights$Variable == "porpag_1")] + porpag_2 * model_2_weights$Importance[which(model_2_weights$Variable == "porpag_2")],
         def_score = dporpag_1 * model_2_weights$Importance[which(model_2_weights$Variable == "dporpag_1")] + dporpag_2 * model_2_weights$Importance[which(model_2_weights$Variable == "dporpag_2")]) %>%
  select(-pick, -porpag_1, -dporpag_1, -porpag_2, -dporpag_2)

three_year_test <- three_year %>% filter(is.na(pick)) %>%
  mutate(off_score = porpag_1 * model_3_weights$Importance[which(model_3_weights$Variable == "porpag_1")] + porpag_2 * model_3_weights$Importance[which(model_3_weights$Variable == "porpag_2") + porpag_3 * model_3_weights$Importance[which(model_3_weights$Variable == "porpag_3")]],
         def_score = dporpag_1 * model_3_weights$Importance[which(model_3_weights$Variable == "dporpag_1")] + dporpag_2 * model_3_weights$Importance[which(model_3_weights$Variable == "dporpag_2")] + dporpag_3 * model_3_weights$Importance[which(model_3_weights$Variable == "dporpag_3")]) %>%
  select(-pick, -porpag_1, -dporpag_1, -porpag_2, -dporpag_2, -porpag_3, -dporpag_3)

four_year_test <- four_year %>% filter(is.na(pick)) %>%
  mutate(off_score = porpag_1 * model_4_weights$Importance[which(model_4_weights$Variable == "porpag_1")] + porpag_2 * model_4_weights$Importance[which(model_4_weights$Variable == "porpag_2") + porpag_3 * model_4_weights$Importance[which(model_4_weights$Variable == "porpag_3")] + porpag_4 * model_4_weights$Importance[which(model_4_weights$Variable == "porpag_4")]],
         def_score = dporpag_1 * model_4_weights$Importance[which(model_4_weights$Variable == "dporpag_1")] + dporpag_2 * model_4_weights$Importance[which(model_4_weights$Variable == "dporpag_2")] + dporpag_3 * model_4_weights$Importance[which(model_4_weights$Variable == "dporpag_3")] + dporpag_4 * model_4_weights$Importance[which(model_4_weights$Variable == "dporpag_4")]) %>%
  select(-pick, -porpag_1, -dporpag_1, -porpag_2, -dporpag_2, -porpag_3, -dporpag_3, -porpag_4, -dporpag_4)

data_2024 <- rbind(one_year_test, two_year_test, three_year_test, four_year_test) %>%
  left_join(draft_players, by = "player") 

confs <- stats_2024 %>% distinct(team, conf)

data_2024$team[which(data_2024$team == "Washington State")] <- "Washington St."
data_2024$team[which(data_2024$team == "Weber State")] <- "Weber St."

data_2024 <- data_2024 %>%
  left_join(confs, by = "team")

make_rating <- function(score) {
  min <- min(score)
  max <- max(score)
  multi <- (score - min)/(max - min)
  return(multi * 60 + 30)
}

data_2024$off_score <- round(make_rating(data_2024$off_score), 1)
data_2024$def_score <- round(make_rating(data_2024$def_score), 1)

players <- data.frame()

for (team in unique(data_2024$team)) {
  each <- get_espn_players(team) 
  players <- rbind(players, each)
}

data_2024 <- left_join(data_2024, players, by = c("player"="displayName")) %>%
  mutate(headshot_link = paste0("https://a.espncdn.com/combiner/i?img=/i/headshots/mens-college-basketball/players/full/", id, ".png")) 

data_2024 <- data_2024 %>% gt_cbb_teams(team, team, logo_height = 30, include_name = FALSE) %>% gt_cbb_conferences(conf, conf, logo_height = 20)

off_leaders <- data_2024 %>%
  arrange(-off_score) %>%
  head(10) %>%
  select(headshot_link, player, team, conf, off_score)

def_leaders <- data_2024 %>%
  arrange(-def_score) %>%
  head(10) %>%
  select(headshot_link, player, team, conf, def_score)

gt_align_caption <- function(left, right) {
  caption <- paste0(
    '<span style="float: left;">', left, '</span>',
    '<span style="float: right;">', right, '</span>'
  )
  return(caption)
}

caption = gt_align_caption("Data from <b>cbbdata</b> & <b>cbbplotR</b>", "Amrit Vignesh | <b>@avsportsanalyst</b>")

off_table <- off_leaders %>% gt() %>% 
  gt_img_rows(columns = headshot_link, height = 40) %>%
  fmt_markdown(team) %>%
  fmt_markdown(conf) %>%
  gt_theme_538() %>%
  cols_align(
    align = "center",
    columns = c(headshot_link, player, team, conf, off_score)
  ) %>%
  gt_hulk_col_numeric(off_score) %>%
  cols_label(
    headshot_link = md(""),
    player = md("**Player**"),
    team = md("**Team**"),
    conf = md("**Conference**"),
    off_score = md("**Score**"),
  ) %>%
  tab_header(
    title = md("**2024 NBA Draft Offensive Leaders**"),
    subtitle = md("*Scores Range From **30** to **90***")
  ) %>%
  tab_source_note(html(caption)) %>%
  opt_align_table_header(align = "center") %>%
  tab_style(
    style = list(
      cell_text(weight = "bold")
    ),
    locations = cells_body(
      columns = c(player, off_score)
    )
  ) 

def_table <- def_leaders %>% gt() %>% 
  gt_img_rows(columns = headshot_link, height = 40) %>%
  fmt_markdown(team) %>%
  fmt_markdown(conf) %>%
  gt_theme_538() %>%
  cols_align(
    align = "center",
    columns = c(headshot_link, player, team, conf, def_score)
  ) %>%
  gt_hulk_col_numeric(def_score) %>%
  cols_label(
    headshot_link = md(""),
    player = md("**Player**"),
    team = md("**Team**"),
    conf = md("**Conference**"),
    def_score = md("**Score**"),
  ) %>%
  tab_header(
    title = md("**2024 NBA Draft Defensive Leaders**"),
    subtitle = md("*Scores Range From **30** to **90***")
  ) %>%
  tab_source_note(html(caption)) %>%
  opt_align_table_header(align = "center") %>%
  tab_style(
    style = list(
      cell_text(weight = "bold")
    ),
    locations = cells_body(
      columns = c(player, def_score)
    )
  ) 

gt_two_column_layout(tables = list(off_table, def_table), "viewer") # manually saved from viewer