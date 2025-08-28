**IMPORTANT**: When executing replace $WORKING_DIR with the Working directory from your environment context

eg: if environment context was..

  <env>
  Working directory: /home/bilbo_baggins/bag_end
  Is directory a git repo: Yes
  Platform: linux
  OS Version: Linux 6.14.0-28-generic
  Today's date: 2025-08-28
  </env>

  if you see the bash instruction

  $WORKING_DIR/claude-scripts/foobar.sh

  you would execute

  /home/bilbo_baggins/bag_end/claude-scripts/foobar.sh