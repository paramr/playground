import React, { PureComponent } from 'react'
import { Root, Routes } from 'react-static'
import { Link } from '@reach/router'

import CssBaseline from '@material-ui/core/CssBaseline'
import AppBar from '@material-ui/core/AppBar'
import Tabs from '@material-ui/core/Tabs'
import Tab from '@material-ui/core/Tab'
import { withStyles } from '@material-ui/core/styles'

// Custom styles
const styles = {
  '@global': {
    img: {
      maxWidth: '100%',
    },
  },
  appBar: {
    flexWrap: 'wrap',
  },
  tabs: {
    width: '100%',
  },
  content: {
    padding: '1rem',
  },
}

class App extends PureComponent {
  render() {
    const { classes } = this.props

    return (
      <Root>
        <div className={classes.container}>
          <CssBaseline />
          <AppBar className={classes.appBar} position="static">
            <nav>
              <Tabs className={classes.tabs} value={false}>
                <Tab component={Link} to="/" label="Home" />
                <Tab component={Link} to="/about" label="About" />
              </Tabs>
            </nav>
          </AppBar>
          <div className={classes.content}>
            <Routes />
          </div>
        </div>
      </Root>
    )
  }
}

const AppWithStyles = withStyles(styles)(App)

export default AppWithStyles
